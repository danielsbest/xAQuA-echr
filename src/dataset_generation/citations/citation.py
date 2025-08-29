"""Citation data model with thread-safe metadata fetching."""
import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from pydantic import BaseModel, Field

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset_generation.citations.judgement_citation_metadata import multilingual_citations
from src.dataset_generation.citations.database_manager import db_manager
from src.dataset_generation.citations.config import config

logger = logging.getLogger(__name__)


class MetadataCache:
    """Thread-safe metadata cache for citations."""
    
    @staticmethod
    def load(url: str) -> Optional[Dict[str, Any]]:
        """Load metadata from cache."""
        try:
            with db_manager.get_metadata_connection(timeout=5.0) as conn:
                cursor = conn.execute(
                    'SELECT data FROM meta_cache WHERE url = ?',
                    (url,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.error(f"Failed to load metadata from cache for {url}: {e}")
        return None
    
    @staticmethod
    def save(url: str, data: Dict[str, Any]) -> None:
        """Save metadata to cache."""
        try:
            with db_manager.get_metadata_connection(timeout=5.0) as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO meta_cache (url, data) VALUES (?, ?)',
                    (url, json.dumps(data))
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save metadata to cache for {url}: {e}")
    
    @staticmethod
    def fetch_with_timeout(url: str, lang_code: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Fetch metadata with timeout."""
        # First check cache
        cached_data = MetadataCache.load(url)
        if cached_data:
            return cached_data
        
        # Fetch with timeout
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(multilingual_citations, url, lang_code)
                data = future.result(timeout=timeout)
                
                # Save to cache if successful
                if data:
                    MetadataCache.save(url, data)
                
                return data
                
        except FuturesTimeoutError:
            logger.error(f"Timeout fetching metadata for {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching metadata for {url}: {e}")
            return None


class Citation(BaseModel):
    """Citation model with multilingual support."""
    
    original_url: str = Field(..., alias="citation_url")
    citation_year: int
    citation_tags: List[str]
    citation_paragraphs: List[int]
    citation_date: str
    multilingual: Dict[str, Dict[str, str]]
    
    class Config:
        """Pydantic configuration."""
        populate_by_name = True
    
    @staticmethod
    def normalize_hudoc_url(url: str) -> str:
        """Normalize HUDOC URLs to standard format.
        
        Examples converted to https://hudoc.echr.coe.int/<lang>?i=<ITEMID>
        Supports variants like:
          https://hudoc.echr.coe.int/eng#{"itemid":["001-210332"]}
          https://hudoc.echr.coe.int/eng#%7B%22itemid%22:[%22001-210332%22]%7D
          https://hudoc.echr.coe.int/eng#{%22itemid%22:[%22001-225328%22]}
        Falls back to original URL if parsing fails.
        """
        # Already normalized
        if '?i=' in url:
            return url
        
        from urllib.parse import urlsplit, unquote
        
        try:
            parts = urlsplit(url)
            # Expect path like /eng
            path_segments = [p for p in parts.path.split('/') if p]
            if not path_segments:
                return url
            lang = path_segments[0]
            
            fragment = parts.fragment
            if not fragment and '#' in url:
                # Fallback (very edge cases)
                fragment = url.split('#', 1)[1]
            if not fragment:
                return url
            
            # Decode any percent-encoding in the fragment
            decoded = unquote(fragment)
            
            # Ensure we have braces for JSON
            if not (decoded.startswith('{') and decoded.endswith('}')):
                decoded = '{' + decoded.strip('{} \n\t') + '}'
            
            try:
                data = json.loads(decoded)
            except json.JSONDecodeError:
                return url
            
            # Try common key variants
            item_ids = (
                data.get('itemid')
                or data.get('ItemId')
                or data.get('ITEMID')
            )
            if isinstance(item_ids, list) and item_ids:
                item_id = item_ids[0]
                if isinstance(item_id, str):
                    return f'https://hudoc.echr.coe.int/{lang}?i={item_id}'
        except Exception:
            pass
        return url
    
    @classmethod
    def from_llm_extraction(
        cls,
        *,
        citation_url: str,
        citation_year: Optional[int],
        citation_tags: List[str],
        citation_paragraphs: List[int],
        lang_code: str,
        timeout: float = 30.0
    ) -> 'Citation':
        """Create Citation with API-fetched metadata."""
        normalized_url = cls.normalize_hudoc_url(citation_url)
        
        multilingual_meta = MetadataCache.fetch_with_timeout(
            normalized_url,
            lang_code,
            timeout=timeout
        )
        
        if not multilingual_meta:
            logger.warning(f"Failed to fetch metadata for {normalized_url}, using empty metadata")
            return cls(
                citation_url=normalized_url,
                citation_year=citation_year or 0,
                citation_tags=citation_tags,
                citation_paragraphs=citation_paragraphs,
                citation_date="",
                multilingual={}
            )
        
        multilingual = multilingual_meta.get("lang_versions", {})
        citation_date = multilingual_meta.get("kpdate", "")
        
        if not citation_year and citation_date:
            try:
                citation_year = int(citation_date[:4])
            except (ValueError, IndexError):
                citation_year = 0
        
        return cls(
            citation_url=normalized_url,
            citation_year=citation_year or 0,
            citation_tags=citation_tags,
            citation_paragraphs=citation_paragraphs,
            citation_date=citation_date,
            multilingual=multilingual
        )
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Citation':
        """Create Citation from JSON data."""
        return cls(
            citation_url=data.get("original_url", data.get("citation_url", "")),
            citation_year=data.get("citation_year", 0),
            citation_tags=data.get("citation_tags", []),
            citation_paragraphs=data.get("citation_paragraphs", []),
            citation_date=data.get("citation_date", ""),
            multilingual=data.get("multilingual", {})
        )
    
    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON."""
        return {
            "original_url": self.original_url,
            "citation_year": self.citation_year,
            "citation_tags": self.citation_tags,
            "citation_paragraphs": self.citation_paragraphs,
            "citation_date": self.citation_date,
            "multilingual": self.multilingual
        }
    
    def citation_string(self, main_lang: Optional[str] = None) -> str:
        """Human-readable representation."""
        main_name = None
        alternative_names = []
        lang_priority = ([main_lang] if main_lang else []) + ['fre', 'eng']
        
        for lang in lang_priority:
            if not main_name and self.multilingual.get(lang):
                main_name = self.multilingual[lang].get('docname', '')
            elif self.multilingual.get(lang):
                alt_name = self.multilingual[lang].get('docname', '')
                if alt_name:
                    alternative_names.append(alt_name)
        
        if not main_name:
            main_name = "Unknown Case"
        
        alternative_name_string = (
            f", Alternative names: {', '.join(alternative_names)}"
            if alternative_names else ''
        )
        
        paragraphs = (
            f"Paragraphs: {', '.join(map(str, self.citation_paragraphs))}"
            if self.citation_paragraphs else "No paragraphs"
        )
        
        tags = f" {' '.join(self.citation_tags)}" if self.citation_tags else ''
        
        return f"{self.citation_year}{tags}, {main_name}, {paragraphs}{alternative_name_string}"
    
    def considered_same(self, other: 'Citation') -> bool:
        """Check if two citations refer to the same case."""
        if self.citation_paragraphs != other.citation_paragraphs:
            return False
        
        own_ids = {
            data.get('id')
            for data in self.multilingual.values()
            if data.get('id')
        }
        
        other_ids = {
            data.get('id')
            for data in other.multilingual.values()
            if data.get('id')
        }
        
        # Check for ID overlap
        return bool(own_ids & other_ids) if (own_ids and other_ids) else False
    
    def __str__(self) -> str:
        """String representation."""
        return self.citation_string()
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"Citation(url={self.original_url}, year={self.citation_year}, paragraphs={self.citation_paragraphs})"


if __name__ == "__main__":
    test_json = {
        "original_url": "https://hudoc.echr.coe.int/fre?i=001-105607",
        "citation_year": 2011,
        "citation_tags": ["MC"],
        "citation_paragraphs": [],
        "citation_date": "2011-07-07T00:00:00",
        "multilingual": {
            "eng": {
                "id": "001-105606",
                "docname": "CASE OF AL-SKEINI AND OTHERS v. THE UNITED KINGDOM"
            },
            "fre": {
                "id": "001-105607",
                "docname": "AFFAIRE AL-SKEINI ET AUTRES c. ROYAUME-UNI"
            },
            "ron": {
                "id": "001-114880",
                "docname": "CASE OF AL-SKEINI AND OTHERS v. THE UNITED KINGDOM - [Romanian Translation]"
            }
        }
    }
    
    citation = Citation.from_json(test_json)
    citation2 = Citation.from_json(test_json)
    citation2.citation_tags = ['ibidem']
    
    print(f"Citation 1: {citation.citation_string()}")
    print(f"Citation 1 (Romanian): {citation.citation_string(main_lang='ron')}")
    print(f"Same citation? {citation.considered_same(citation2)}")
    
    db_manager.close()