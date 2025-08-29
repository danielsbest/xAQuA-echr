"""Thread-safe judgement fetching with proper resource management."""
import json
import re
import logging
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from contextlib import contextmanager

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset_generation.citations import judgement_parser
from src.dataset_generation.citations.database_manager import db_manager
from src.dataset_generation.citations.config import config

logger = logging.getLogger("md_citations_extraction_logs.judgement_fetching")

SCRIPT_VERSION = "0.8"


class RequestSession:
    """HTTP session with retry logic and connection pooling."""
    
    def __init__(self):
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=50
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get(self, url: str, timeout: Optional[Tuple[float, float]] = None) -> requests.Response:
        """Make GET request with default timeout."""
        if timeout is None:
            timeout = (
                config.threading.web_fetch_connect_timeout,
                config.threading.web_fetch_read_timeout
            )
        return self.session.get(url, timeout=timeout)
    
    def close(self):
        """Close the session."""
        self.session.close()


class CaseLockManager:
    """Manages per-case locks with automatic cleanup."""
    
    def __init__(self, max_locks: int = 1000):
        self._locks: Dict[str, threading.RLock] = {}
        self._lock_refs: Dict[str, weakref.ref] = {}
        self._access_times: Dict[str, float] = {}
        self._master_lock = threading.Lock()
        self._max_locks = max_locks
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    @contextmanager
    def get_lock(self, case_id: str):
        """Get or create a lock for a case."""
        with self._master_lock:
            # Periodic cleanup
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_locks()
            
            # Get or create lock
            if case_id not in self._locks:
                if len(self._locks) >= self._max_locks:
                    self._cleanup_old_locks(force=True)
                
                lock = threading.RLock()
                self._locks[case_id] = lock
                self._lock_refs[case_id] = weakref.ref(lock)
            
            lock = self._locks[case_id]
            self._access_times[case_id] = time.time()
        
        # Acquire lock outside of master lock to avoid deadlock
        with lock:
            yield
    
    def _cleanup_old_locks(self, force: bool = False):
        """Remove old or unused locks."""
        current_time = time.time()
        max_age = 3600  # 1 hour
        
        to_remove = []
        
        for case_id, access_time in list(self._access_times.items()):
            # Remove if old or if weak reference is dead
            if (current_time - access_time > max_age or
                self._lock_refs[case_id]() is None or
                (force and len(to_remove) < len(self._locks) // 2)):
                to_remove.append(case_id)
        
        for case_id in to_remove:
            self._locks.pop(case_id, None)
            self._lock_refs.pop(case_id, None)
            self._access_times.pop(case_id, None)
        
        self._last_cleanup = current_time
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old locks")


class JudgementFetcher:
    """Thread-safe judgement fetcher with improved resource management."""
    
    def __init__(self):
        self._lock_manager = CaseLockManager()
        self._http_session = RequestSession()
        self._closed = False
        self._fetch_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'mongo_fetches': 0,
            'web_fetches': 0,
            'failures': 0
        }
        self._stats_lock = threading.Lock()
    
    def _update_stats(self, stat_name: str):
        """Update fetch statistics."""
        with self._stats_lock:
            self._fetch_stats[stat_name] += 1
    
    def _fetch_from_mongo(self, case_id: str) -> Optional[Dict]:
        """Fetch judgement from MongoDB."""
        try:
            collection = db_manager.get_mongo_collection("echr_documents")
            if collection is None:
                return None
            
            query = {"_id": case_id, "isplaceholder": {"$ne": True}}
            document = collection.find_one(query)
            
            if document:
                self._update_stats('mongo_fetches')
                logger.debug(f"Fetched {case_id} from MongoDB")
            
            return document
            
        except Exception as e:
            logger.error(f"MongoDB fetch error for {case_id}: {e}")
            return None
    
    def _fetch_from_web(self, case_id: str) -> Optional[Dict]:
        """Fetch judgement from HUDOC website."""
        try:
            url = f"https://hudoc.echr.coe.int/app/conversion/docx/html/body?library=ECHR&id={case_id}"
            response = self._http_session.get(url)
            response.raise_for_status()
            
            html_content = response.text
            
            # Parse document name
            soup = BeautifulSoup(html_content, "html.parser")
            docname_tag = soup.find('p', class_='docname')
            docname = docname_tag.get_text(strip=True) if docname_tag else case_id
            
            self._update_stats('web_fetches')
            logger.info(f"Fetched {case_id} from web")
            
            return {"html": html_content, "docname": docname}
            
        except requests.RequestException as e:
            logger.error(f"Web fetch error for {case_id}: {e}")
            self._update_stats('failures')
            return None
    
    def _check_cache(self, case_id: str, paragraph_number: Optional[int] = None) -> Optional[Union[str, bool]]:
        """Check if case/paragraph exists in cache."""
        try:
            with db_manager.get_paragraphs_connection(timeout=5.0) as conn:
                if paragraph_number is not None:
                    # Check for specific paragraph
                    cursor = conn.execute(
                        'SELECT paragraph_text FROM judgements WHERE case_id = ? AND paragraph_number = ?',
                        (case_id, paragraph_number)
                    )
                    row = cursor.fetchone()
                    if row:
                        self._update_stats('cache_hits')
                        return row[0]
                else:
                    # Check if case exists
                    cursor = conn.execute(
                        'SELECT 1 FROM judgements WHERE case_id = ? LIMIT 1',
                        (case_id,)
                    )
                    if cursor.fetchone():
                        self._update_stats('cache_hits')
                        return True
            
            return None
            
        except Exception as e:
            logger.error(f"Cache check error for {case_id}: {e}")
            return None
    
    def _save_to_cache(self, case_id: str, docname: str, paragraphs: Dict[int, str]):
        """Save paragraphs to cache."""
        if not paragraphs:
            return
        
        rows = [
            (case_id, docname, num, text, SCRIPT_VERSION)
            for num, text in paragraphs.items()
        ]
        
        try:
            with db_manager.get_paragraphs_connection(timeout=10.0) as conn:
                conn.executemany(
                    '''INSERT OR IGNORE INTO judgements 
                       (case_id, case_name, paragraph_number, paragraph_text, recently_parsed)
                       VALUES (?, ?, ?, ?, ?)''',
                    rows
                )
                conn.commit()
                logger.debug(f"Saved {len(paragraphs)} paragraphs for {case_id}")
                
        except Exception as e:
            logger.error(f"Failed to save paragraphs for {case_id}: {e}")
    
    def fetch_judgement(
        self,
        case_id: str,
        paragraph_number: Optional[int] = None
    ) -> Union[str, bool, None]:
        """Fetch judgement with proper locking and caching. 
        
        Returns:
            str: Paragraph text if paragraph_number provided and found
            'Paragraph missing': If case cached but paragraph not present
            True: If case cached/fetched successfully
            False/None: On failure
        """
        if self._closed:
            raise RuntimeError("JudgementFetcher is closed")
        
        # Fast path: check cache first
        cached = self._check_cache(case_id, paragraph_number)
        if cached:
            return cached
        
        # Check if we're looking for a missing paragraph in cached case
        if paragraph_number is not None and self._check_cache(case_id):
            return 'Paragraph missing'
        
        # Need to fetch - use per-case lock
        with self._lock_manager.get_lock(case_id):
            # Double-check after acquiring lock
            cached = self._check_cache(case_id, paragraph_number)
            if cached:
                return cached
            
            if paragraph_number is not None and self._check_cache(case_id):
                return 'Paragraph missing'
            
            # Fetch from external sources
            self._update_stats('cache_misses')
            logger.info(f"Cache miss for {case_id}, fetching...")
            
            # Try MongoDB first
            document = self._fetch_from_mongo(case_id)
            
            # Fallback to web if needed
            if not document or document.get('html') == '{"Message":"An error has occurred."}':
                logger.info(f"MongoDB fetch failed for {case_id}, trying web...")
                document = self._fetch_from_web(case_id)
            
            if not document or not document.get('html'):
                logger.error(f"Failed to fetch {case_id} from all sources")
                self._update_stats('failures')
                return False
            
            # Parse paragraphs
            try:
                paragraphs = judgement_parser.parse_judgement_paragraphs(document['html'])
                if not paragraphs:
                    logger.error(f"No paragraphs parsed for {case_id}")
                    return False
                
                # Save to cache
                self._save_to_cache(case_id, document.get('docname', case_id), paragraphs)
                
                # Return requested data
                if paragraph_number is not None:
                    return paragraphs.get(paragraph_number, 'Paragraph missing')
                return True
                
            except Exception as e:
                logger.error(f"Failed to parse paragraphs for {case_id}: {e}")
                self._update_stats('failures')
                return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get fetch statistics."""
        with self._stats_lock:
            return dict(self._fetch_stats)
    
    def fix_missing_case_names(self):
        """Fix missing case names using metadata cache."""
        logger.info("Fixing missing case names...")
        
        try:
            # Find cases with missing names
            with db_manager.get_paragraphs_connection() as conn:
                cursor = conn.execute('''
                    SELECT DISTINCT case_id 
                    FROM judgements 
                    WHERE case_name = case_id OR case_name IS NULL
                ''')
                case_ids = [row[0] for row in cursor.fetchall()]
            
            if not case_ids:
                logger.info("No missing case names found")
                return
            
            logger.info(f"Found {len(case_ids)} cases with missing names")
            
            # Get metadata and update names
            fixed_count = 0
            
            with db_manager.get_metadata_connection() as meta_conn, db_manager.get_paragraphs_connection() as para_conn:
                id_docname_regex_cache = {}
                for case_id in case_ids:
                    pattern = f'%"id": "{case_id}"%'
                    cursor = meta_conn.execute(
                        'SELECT data FROM meta_cache WHERE data LIKE ? LIMIT 1',
                        (pattern,)
                    )
                    row = cursor.fetchone()
                    if not row:
                        continue
                    raw_json = row[0]
                    # Compile regex once (cache by case_id length maybe unnecessary but cheap)
                    if case_id not in id_docname_regex_cache:
                        id_docname_regex_cache[case_id] = re.compile(
                            r'"id"\s*:\s*"' + re.escape(case_id) + r'"\s*,\s*"docname"\s*:\s*"(.*?)"'
                        )
                    match = id_docname_regex_cache[case_id].search(raw_json)
                    case_name = match.group(1).strip() if match else None
                    if not case_name:
                        # Fallback to JSON parse only if regex failed
                        try:
                            metadata = json.loads(raw_json)
                            lang_versions = metadata.get('lang_versions', {}) or {}
                            for lang in ('eng', 'fre'):
                                if lang in lang_versions:
                                    case_name = lang_versions[lang].get('docname')
                                    if case_name:
                                        break
                            if not case_name and lang_versions:
                                case_name = next(iter(lang_versions.values())).get('docname')
                            if case_name:
                                case_name = case_name.strip()
                        except Exception:
                            continue
                    if case_name and case_name != case_id:
                        para_conn.execute(
                            'UPDATE judgements SET case_name = ? WHERE case_id = ?',
                            (case_name, case_id)
                        )
                        fixed_count += 1
                        if fixed_count % 100 == 0:
                            logger.info(f"Fixed {fixed_count}/{len(case_ids)} case names...")
                para_conn.commit()
            
            logger.info(f"Successfully fixed {fixed_count} case names")
            
        except Exception as e:
            logger.error(f"Error fixing case names: {e}")
    
    def export_to_csv(self, csv_path: str = 'data/echr_case_paragraphs.csv'):
        """Export cache to CSV with proper ordering."""
        csv_full_path = project_root / csv_path
        logger.info(f"Exporting to CSV: {csv_full_path}")
        
        try:
            # Check for existing CSV and get existing versions
            existing_versions = set()
            existing_df = None
            if csv_full_path.exists():
                existing_df = pd.read_csv(csv_full_path)
                if 'recently_parsed' in existing_df.columns:
                    # Get all unique versions already in the CSV
                    existing_versions = set(existing_df['recently_parsed'].unique())
                    logger.info(f"Found existing versions in CSV: {sorted(existing_versions)}")
                logger.info(f"Preserving {len(existing_df)} existing rows")
            
            # Get all data from database
            with db_manager.get_paragraphs_connection() as conn:
                query = '''
                    SELECT case_id, case_name, paragraph_number, paragraph_text, recently_parsed
                    FROM judgements
                    ORDER BY case_id, paragraph_number
                '''
                db_df = pd.read_sql_query(query, conn)
            
            # Get all unique versions from database
            db_versions = set(db_df['recently_parsed'].unique()) if not db_df.empty else set()
            logger.info(f"Found versions in database: {sorted(db_versions)}")
            
            # Find new versions not already in CSV
            new_versions = db_versions - existing_versions
            logger.info(f"New versions to add: {sorted(new_versions)}")
            
            # Start with existing CSV data (if any)
            dfs_to_concat = []
            if existing_df is not None and not existing_df.empty:
                dfs_to_concat.append(existing_df)
            
            # Add new versions in sorted order
            for version in sorted(new_versions):
                version_df = db_df[db_df['recently_parsed'] == version].copy()
                if not version_df.empty:
                    logger.info(f"Adding {len(version_df)} rows for version {version}")
                    dfs_to_concat.append(version_df)
            
            if dfs_to_concat:
                final_df = pd.concat(dfs_to_concat, ignore_index=True)
                final_df.to_csv(csv_full_path, index=False)
                logger.info(f"Exported {len(final_df)} rows to {csv_full_path}")
                
                # Log final version distribution
                version_counts = final_df['recently_parsed'].value_counts().sort_index()
                logger.info(f"Final version distribution: {version_counts.to_dict()}")
            else:
                logger.warning("No data to export")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def close(self):
        """Clean up resources."""
        if not self._closed:
            self._closed = True
            self._http_session.close()
            
            # Log final statistics
            stats = self.get_stats()
            logger.info(f"JudgementFetcher statistics: {stats}")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.close()


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the refactored JudgementFetcher
    sample_case_id = "001-68183"
    fetcher = JudgementFetcher()
    
    try:
        print(f"Testing fetch for case {sample_case_id}, paragraph 5...")
        
        # First fetch (cache miss)
        start = time.time()
        result = fetcher.fetch_judgement(sample_case_id, paragraph_number=5)
        elapsed = time.time() - start
        
        if isinstance(result, str) and 'missing' not in result:
            print(f"✓ Fetched paragraph 5 in {elapsed:.2f}s: {result[:100]}...")
        else:
            print(f"✗ Failed to fetch paragraph 5: {result}")
        
        # Second fetch (cache hit)
        start = time.time()
        result = fetcher.fetch_judgement(sample_case_id, paragraph_number=5)
        elapsed = time.time() - start
        
        if isinstance(result, str):
            print(f"✓ Cache hit in {elapsed:.2f}s: {result[:100]}...")
        else:
            print(f"✗ Unexpected cache result: {result}")
        
        # Show statistics
        print(f"\nFetch statistics: {fetcher.get_stats()}")
        
    finally:
        fetcher.close()
        db_manager.close()