import sqlite3
import hashlib
import time
import os
from typing import Optional, Any
from typing import List, Dict, Tuple

class Cache:
    def __init__(self, db_path: str = "data/llm_cache.db"):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self) -> None:
        """
        Ensures the cache table exists and sets the journal mode to WAL.
        This is called only once when the Cache object is instantiated.
        """
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash TEXT,
                model_name TEXT,
                temperature REAL,
                prompt TEXT,
                response_content TEXT,
                timestamp REAL,
                PRIMARY KEY (prompt_hash, model_name, temperature)
            )
            """)
            conn.commit()

    def _get_prompt_hash(self, prompt: str, **kwargs: Any) -> str:
        system_prompt = kwargs.get('system_prompt')
        prompt_to_hash = system_prompt + prompt if system_prompt else prompt
        return hashlib.sha256(prompt_to_hash.encode()).hexdigest()

    def get(self, prompt: str, model_name: str, temperature: float, **kwargs: Any) -> Optional[str]:

        prompt_hash = self._get_prompt_hash(prompt, **kwargs)

        # Connect in read-only mode to allow concurrent writes.
        db_uri = f"file:{os.path.abspath(self.db_path)}?mode=ro"
        
        with sqlite3.connect(db_uri, uri=True, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT response_content FROM llm_cache WHERE prompt_hash = ? AND model_name = ? AND temperature = ?',
                (prompt_hash, model_name, temperature)
            )
            result = cursor.fetchone()
            if result:
                print("Cache hit")
                return result[0]
        
        print("Cache miss")
        return None

    def get_many(self, prompts: List[Dict[str, Any]], model_name: str) -> Tuple[Dict[tuple, str], List[Dict[str, Any]]]:
        """
        Tries to get multiple prompts from the cache in a single, efficient query.
        This correctly handles prompts with different temperatures.

        Returns a tuple: (cached_results, uncached_prompts)
        - cached_results is a dict of {(prompt_hash, model_name, temperature): response_content}
        - uncached_prompts is a list of prompt dicts that were not found.
        """
        if not prompts:
            return {}, []

        # Prepare unique composite keys for all prompts to be checked.
        prompts_by_key = {}
        for p_info in prompts:
            prompt = p_info['prompt']
            temperature = p_info.get('temperature', 1.0)
            prompt_hash = self._get_prompt_hash(**p_info)
            request_key = (prompt_hash, model_name, float(temperature))
            if request_key not in prompts_by_key:
                prompts_by_key[request_key] = p_info

        # Connect to DB and fetch all matching entries, batching to avoid SQLite depth limits
        db_uri = f"file:{os.path.abspath(self.db_path)}?mode=ro"
        cached_results = {}
        
        # Split into batches to avoid SQLite expression tree depth limit (1000)
        batch_size = 200
        keys_list = list(prompts_by_key.keys())
        
        with sqlite3.connect(db_uri, uri=True, timeout=30) as conn:
            cursor = conn.cursor()

            for i in range(0, len(keys_list), batch_size):
                batch_keys = keys_list[i:i + batch_size]
                
                conditions = []
                params = []
                for ph, mn, temp in batch_keys:
                    conditions.append("(prompt_hash = ? AND model_name = ? AND temperature = ?)")
                    params.extend([ph, mn, temp])

                if not conditions:
                    continue

                query = f"SELECT prompt_hash, model_name, temperature, response_content FROM llm_cache WHERE {' OR '.join(conditions)}"
                cursor.execute(query, params)

                for res_hash, res_model, res_temp, res_content in cursor.fetchall():
                    result_key = (res_hash, res_model, float(res_temp))
                    cached_results[result_key] = res_content

        # Only run prompts which were not found in the cache.
        found_keys = set(cached_results.keys())
        uncached_prompts = [p_info for key, p_info in prompts_by_key.items() if key not in found_keys]

        if cached_results:
            print(f"Cache hit for {len(cached_results)} items.")
        if uncached_prompts:
            print(f"Cache miss for {len(uncached_prompts)} items.")

        return cached_results, uncached_prompts

    def set(self, prompt: str, model_name: str, response_content: str, temperature: float, **kwargs: Any) -> None:
        """Sets a single prompt-response pair in the cache."""
        prompt_info = {'prompt': prompt, 'temperature': temperature, **kwargs}
        self.set_many([prompt_info], [response_content], model_name)

    def set_many(self, prompts: List[Dict[str, Any]], responses: list, model_name: str) -> None:
        """Stores multiple prompt-response pairs in the cache in a single transaction."""
        if not prompts:
            return

        all_kwargs_keys = set()
        rows_to_insert = []
        
        for p_info, result in zip(prompts, responses):
            # Handle cases where result is a tuple (response, thoughts) (deepseek r1)
            if isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], str):
                response_content, thoughts = result
                p_info['thoughts'] = thoughts
            else:
                response_content = result

            prompt = p_info['prompt']
            temperature = p_info.get('temperature', 1.0)
            prompt_hash = self._get_prompt_hash(**p_info)

            # Collect all dynamic kwargs
            kwargs = {k: v for k, v in p_info.items() if k not in ['prompt', 'temperature']}
            all_kwargs_keys.update(kwargs.keys())

            row_data = {
                'prompt_hash': prompt_hash, 'model_name': model_name, 'temperature': temperature,
                'response_content': response_content, 'prompt': prompt, 'timestamp': time.time(),
                **kwargs
            }
            rows_to_insert.append(row_data)

        # perform schema changes and insertion
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.cursor()
            
            cursor.execute("PRAGMA table_info(llm_cache)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            columns_to_add = all_kwargs_keys - existing_columns
            for col in sorted(list(columns_to_add)):
                cursor.execute(f'ALTER TABLE llm_cache ADD COLUMN "{col}" TEXT')

            final_columns = sorted(list(existing_columns | all_kwargs_keys | {'prompt_hash', 'model_name', 'temperature', 'response_content', 'prompt', 'timestamp'}))
            columns_str = ', '.join(f'"{c}"' for c in final_columns)
            placeholders = ', '.join(['?'] * len(final_columns))
            
            final_rows = [tuple(row.get(col) for col in final_columns) for row in rows_to_insert]

            if final_rows:
                cursor.executemany(f'INSERT OR REPLACE INTO llm_cache ({columns_str}) VALUES ({placeholders})', final_rows)
            conn.commit()