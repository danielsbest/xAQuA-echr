import json
import random
import sqlite3
import sys
import os
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset_generation.citations.citation import Citation


# logging
import logging
import logging.config
logging_config_path = project_root / 'src' / 'logging' / 'logging.json'
with open(logging_config_path, 'rt') as f:
        config = json.load(f)

for handler_name, handler_config in config.get('handlers', {}).items():
    if 'filename' in handler_config:
        relative_path = Path(handler_config['filename'])
        absolute_path = project_root / relative_path
        handler_config['filename'] = str(absolute_path)

logging.config.dictConfig(config)
logger = logging.getLogger("generation_logs")

def select_random_sample(guides_df: pd.DataFrame, grouping_method: str, num_guides: int = 15, num_groups: int = 10, random_seed:int = 112) -> pd.DataFrame:
    unique_guide_ids = guides_df["guide_id"].unique()
    random.seed(random_seed)
    selected_guide_ids = random.sample(
        list(unique_guide_ids), num_guides
    )
    
    filtered_df = guides_df[guides_df["guide_id"].isin(selected_guide_ids)]

    sampled_rows = []
    for guide_id in selected_guide_ids:
        guide_data = filtered_df[filtered_df["guide_id"] == guide_id]
        unique_groups = guide_data[grouping_method].unique()
        
        # Sample up to num_groups (or all if fewer available)
        num_groups = min(num_groups, len(unique_groups))
        selected_groups = random.sample(list(unique_groups), num_groups)
        
        # Get all rows for the selected groups
        guide_sampled = guide_data[guide_data[grouping_method].isin(selected_groups)]
        sampled_rows.append(guide_sampled)
    
    return pd.concat(sampled_rows, ignore_index=True)

def numbered_string(strings: list[str]):
    return "\n".join(f"[{i+1}]: {s}" for i, s in enumerate(strings))

def filter_duplicates(*citation_lists: list[Citation]) -> list[Citation]:
    """
    merges multiple lists of Citation objects into a single list,
    removes duplicates
    """
    merged: list[Citation] = []
    for citations in citation_lists:
        for cit in citations:
            if not any(existing.considered_same(cit) for existing in merged):
                merged.append(cit)
    return merged


def write_to_db(result_row: dict, output_db_path: str):
    """Save a single result row to the database immediately after processing"""
    conn = None
    try:
        conn = sqlite3.connect(output_db_path, timeout=30.0)
        
        conn.execute("PRAGMA journal_mode=WAL")

        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS question_generations (
                guide_id TEXT,
                group_id INTEGER,
                question TEXT,
                answer TEXT,
                paragraphs TEXT,
                paragraph_nums TEXT,
                num_paragraphs INTEGER,
                grouping_method TEXT,
                model_name TEXT,
                lang_code TEXT,
                PRIMARY KEY (guide_id, group_id, grouping_method, model_name)
            )
        ''')
        
        cursor.execute('''
            INSERT OR REPLACE INTO question_generations 
            (guide_id, group_id, question, answer, paragraphs, paragraph_nums, 
             num_paragraphs, grouping_method, model_name, lang_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_row['guide_id'],
            result_row['group_id'],
            result_row['question'],
            result_row['answer'],
            result_row['paragraphs'],
            result_row['paragraph_nums'],
            result_row['num_paragraphs'],
            result_row['grouping_method'],
            result_row['model_name'],
            result_row['lang_code']
        ))
        
        conn.commit()
        logger.info(f"Saved to database: guide {result_row['guide_id']}, group {result_row['group_id']}")
    except sqlite3.Error as e:
        logger.error(f"Database error for guide_id={result_row.get('guide_id')}, group_id={result_row.get('group_id')}: {e}. Skipping item.")
    finally:
        if conn:
            conn.close()