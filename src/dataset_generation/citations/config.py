"""Configuration management for citation extraction system."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).resolve().parents[3]


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    
    # SQLite settings
    sqlite_timeout: float = 30.0
    sqlite_busy_timeout: int = 30000  # milliseconds
    sqlite_page_size: int = 4096
    sqlite_cache_size: int = -2000  # negative means KB
    
    # MongoDB settings
    mongo_uri: Optional[str] = None
    mongo_connection_timeout: int = 10000  # ms
    mongo_server_selection_timeout: int = 10000  # ms
    mongo_socket_timeout: int = 60000  # ms
    mongo_max_pool_size: int = 50
    mongo_min_pool_size: int = 10
    
    # Database paths
    metadata_cache_path: Path = project_root / 'data' / 'judgement_metadata_cache.db'
    paragraphs_cache_path: Path = project_root / 'data' / 'judgement_paragraphs_cache.db'
    citations_db_path: Path = project_root / 'data' / 'extracted_citations.db'
    
    def __post_init__(self):
        """Initialize MongoDB URI from environment variables."""
        mongo_username = os.getenv('MONGO_DB_USERNAME')
        mongo_password = os.getenv('MONGO_DB_PWD')
        if mongo_username and mongo_password:
            self.mongo_uri = f"mongodb://{mongo_username}:{mongo_password}@f27se1.in.tum.de:27017/echr"


@dataclass
class ThreadingConfig:
    """Threading and concurrency configuration."""
    
    max_workers: int = int(os.getenv('MAX_WORKERS', '16'))
    thread_pool_timeout: float = 300.0  # seconds
    
    # Task timeouts
    llm_inference_timeout: float = float(os.getenv('LLM_INFERENCE_TIMEOUT', '180'))
    web_fetch_connect_timeout: float = 10.0
    web_fetch_read_timeout: float = 60.0
    api_call_timeout: float = 30.0
    
    # Watchdog settings
    watchdog_interval: float = float(os.getenv('WATCHDOG_INTERVAL', '300'))
    stuck_threshold: float = float(os.getenv('STUCK_THRESHOLD', '900'))
    max_watchdog_cycles: int = int(os.getenv('MAX_WATCHDOG_CYCLES', '4'))
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_file_backup_count: int = 5


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.threading = ThreadingConfig()
        self.logging = LoggingConfig()
        
        # Ensure all database directories exist
        for path in [self.database.metadata_cache_path,
                     self.database.paragraphs_cache_path,
                     self.database.citations_db_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load(cls) -> 'Config':
        """Load configuration from environment and defaults."""
        return cls()


# Global configuration instance
config = Config.load()