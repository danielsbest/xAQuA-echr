"""Thread-safe database connection management."""
import sqlite3
import threading
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from queue import Queue
import time

from pymongo import MongoClient, errors
from pymongo.collection import Collection
from pymongo.database import Database

from .config import config

logger = logging.getLogger(__name__)


class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, db_path: Path, pool_size: int = 10):
        self.db_path = db_path
        self.pool_size = pool_size
        self._connections: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._closed = False
        
        # Initialize the pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self._connections.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimized settings."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=config.database.sqlite_timeout,
            check_same_thread=False,
            isolation_level='DEFERRED'
        )
        
        # Optimize SQLite settings
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute(f'PRAGMA busy_timeout={config.database.sqlite_busy_timeout}')
        conn.execute(f'PRAGMA page_size={config.database.sqlite_page_size}')
        conn.execute(f'PRAGMA cache_size={config.database.sqlite_cache_size}')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA temp_store=MEMORY')
        conn.execute('PRAGMA mmap_size=30000000000')
        
        return conn
    
    @contextmanager
    def get_connection(self, timeout: float = 30.0):
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        start_time = time.time()
        
        while conn is None:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    raise TimeoutError("Failed to get connection from pool")
                
                conn = self._connections.get(timeout=min(1.0, remaining_time))
                
                # Test if connection is still valid
                try:
                    conn.execute("SELECT 1")
                except sqlite3.Error:
                    # Connection is broken, create a new one
                    logger.warning("Broken connection detected, creating new one")
                    conn.close()
                    conn = self._create_connection()
                
                yield conn
                
            finally:
                if conn:
                    # Return connection to pool
                    try:
                        self._connections.put_nowait(conn)
                    except:
                        # Pool is full, close the connection
                        conn.close()
    
    def close(self):
        """Close all connections in the pool."""
        with self._lock:
            if self._closed:
                return
            
            self._closed = True
            
            while not self._connections.empty():
                try:
                    conn = self._connections.get_nowait()
                    conn.close()
                except:
                    pass


class MongoConnectionPool:
    """MongoDB connection pool wrapper."""
    
    def __init__(self):
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        self._lock = threading.Lock()
        self._closed = False
    
    def _ensure_connection(self) -> bool:
        """Ensure MongoDB connection is established."""
        if self._client is not None:
            try:
                # Ping to check if connection is alive
                self._client.admin.command('ping')
                return True
            except:
                logger.warning("MongoDB connection lost, reconnecting...")
                self._client = None
        
        if not config.database.mongo_uri:
            logger.error("MongoDB URI not configured")
            return False
        
        with self._lock:
            if self._client is not None:
                return True
            
            try:
                self._client = MongoClient(
                    config.database.mongo_uri,
                    serverSelectionTimeoutMS=config.database.mongo_server_selection_timeout,
                    connectTimeoutMS=config.database.mongo_connection_timeout,
                    socketTimeoutMS=config.database.mongo_socket_timeout,
                    maxPoolSize=config.database.mongo_max_pool_size,
                    minPoolSize=config.database.mongo_min_pool_size,
                    retryWrites=True,
                    retryReads=True
                )
                
                # Test connection
                self._client.admin.command('ping')
                self._database = self._client['new_echr']
                
                logger.info("MongoDB connection established")
                return True
                
            except errors.ConnectionFailure as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                self._client = None
                return False
    
    def get_collection(self, name: str) -> Optional[Collection]:
        """Get a MongoDB collection."""
        if not self._ensure_connection():
            return None
        
        if self._database is None:
            return None
        
        return self._database[name]
    
    def close(self):
        """Close MongoDB connection."""
        with self._lock:
            if self._closed:
                return
            
            self._closed = True
            
            if self._client:
                self._client.close()
                self._client = None
                self._database = None


class DatabaseManager:
    """Centralized database manager for all database operations."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        self.metadata_pool = SQLiteConnectionPool(config.database.metadata_cache_path)
        self.paragraphs_pool = SQLiteConnectionPool(config.database.paragraphs_cache_path)
        self.citations_pool = SQLiteConnectionPool(config.database.citations_db_path)
        self.mongo_pool = MongoConnectionPool()
        
        self._initialize_schemas()
        self._initialized = True
    
    def _initialize_schemas(self):
        """Initialize database schemas."""
        # Metadata cache schema
        with self.metadata_pool.get_connection() as conn:
            # Check if table exists and has the created_at column
            cursor = conn.execute("PRAGMA table_info(meta_cache)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if not columns:
                # Table doesn't exist, create it with all columns
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS meta_cache (
                        url TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_meta_cache_created ON meta_cache(created_at)')
            elif 'created_at' not in column_names:
                # Table exists but missing created_at column, add it without default
                conn.execute('ALTER TABLE meta_cache ADD COLUMN created_at TIMESTAMP')
                # Update existing rows with current timestamp
                conn.execute('UPDATE meta_cache SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL')
                try:
                    conn.execute('CREATE INDEX IF NOT EXISTS idx_meta_cache_created ON meta_cache(created_at)')
                except sqlite3.OperationalError:
                    pass  # Index might already exist
            conn.commit()
        
        # Paragraphs cache schema
        with self.paragraphs_pool.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS judgements (
                    case_id TEXT NOT NULL,
                    case_name TEXT,
                    paragraph_number INTEGER NOT NULL,
                    paragraph_text TEXT,
                    recently_parsed TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (case_id, paragraph_number)
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_judgements_case ON judgements(case_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_judgements_parsed ON judgements(recently_parsed)')
            conn.commit()
        
        # Citations database schema
        with self.citations_pool.get_connection() as conn:
            # Check if table exists and what columns it has
            cursor = conn.execute("PRAGMA table_info(citation_extractions)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if not columns:
                # Table doesn't exist, create it with all columns
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS citation_extractions (
                        guide_id TEXT NOT NULL,
                        paragraph_id TEXT NOT NULL,
                        paragraph_text TEXT,
                        sentences TEXT,
                        citations TEXT,
                        errors TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (guide_id, paragraph_id)
                    )
                ''')
            else:
                # Table exists, check for missing columns
                if 'created_at' not in column_names:
                    conn.execute('ALTER TABLE citation_extractions ADD COLUMN created_at TIMESTAMP')
                    conn.execute('UPDATE citation_extractions SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL')
                if 'updated_at' not in column_names:
                    conn.execute('ALTER TABLE citation_extractions ADD COLUMN updated_at TIMESTAMP')
                    conn.execute('UPDATE citation_extractions SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_citations_guide ON citation_extractions(guide_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_citations_errors ON citation_extractions(errors)')
            conn.commit()
    
    @contextmanager
    def get_metadata_connection(self, timeout: float = 30.0):
        """Get metadata database connection."""
        with self.metadata_pool.get_connection(timeout) as conn:
            yield conn
    
    @contextmanager
    def get_paragraphs_connection(self, timeout: float = 30.0):
        """Get paragraphs database connection."""
        with self.paragraphs_pool.get_connection(timeout) as conn:
            yield conn
    
    @contextmanager
    def get_citations_connection(self, timeout: float = 30.0):
        """Get citations database connection."""
        with self.citations_pool.get_connection(timeout) as conn:
            yield conn
    
    def get_mongo_collection(self, name: str) -> Optional[Collection]:
        """Get MongoDB collection."""
        return self.mongo_pool.get_collection(name)
    
    def execute_with_retry(
        self,
        connection_getter,
        query: str,
        params: Optional[Tuple] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[List[Tuple]]:
        """Execute a query with retry logic."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                with connection_getter() as conn:
                    cursor = conn.execute(query, params or ())
                    return cursor.fetchall()
            
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Database query failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Database query failed after {max_retries} attempts: {e}")
        
        raise last_error
    
    def close(self):
        """Close all database connections."""
        self.metadata_pool.close()
        self.paragraphs_pool.close()
        self.citations_pool.close()
        self.mongo_pool.close()
    
    def __del__(self):
        """Ensure connections are closed on deletion."""
        if hasattr(self, '_initialized') and self._initialized:
            self.close()


# Global database manager instance
db_manager = DatabaseManager()