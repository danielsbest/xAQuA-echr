
import sys
import os
import pandas as pd
import sqlite3
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrievers.qwen3 import QwenEmbeddings
from src.constants import ECHR_QA_CSV_PATH
from src.column import Column

TABLE_NAME = "query_embeddings"

INSTRUCTIONS = {
    "no_instruct": None,
    'base': "Given a query, retrieve relevant documents",
    'base_ron': "Given a Romanian query, retrieve relevant documents",
    'ron_base_ron': "Pentru o interogare în limba română, recuperează documentele relevante",
    
    'LegalBenchCorporateLobbying': "Retrieval the relevant passage for the given query",
    'LegalBenchCorporateLobbying_ron': "Retrieval the relevant passage for the given query in Romanian",

    'echr_retrieve': "Retrieve European Court of Human Rights (ECHR) judgement paragraphs relevant to a general European human rights question",
    'echr_retrieve_ron': "Retrieve European Court of Human Rights (ECHR) judgement paragraphs relevant to a general European human rights question in Romanian",
    'ron_echr_retrieve_ron': "Recuperează paragrafe din hotărârile Curții Europene a Drepturilor Omului (CEDO) relevante pentru o întrebare generală privind drepturile omului europene",
}

def create_db_and_table(db_path: str):
    """Creates the SQLite database and the query_embeddings table if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            query TEXT PRIMARY KEY,
            embedding TEXT NOT NULL
        )
        """)
        conn.commit()

def get_existing_queries(db_path: str) -> set:
    """Fetches the set of queries that are already in the database."""
    if not os.path.exists(db_path):
        return set()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT query FROM {TABLE_NAME}")
            return {row[0] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            return set()

def main():
    """Reads queries, generates embeddings for a specific instruction, and stores them in a dedicated SQLite database."""
    parser = argparse.ArgumentParser(description="Generate and store Qwen embeddings for queries using different instructions.")
    parser.add_argument(
        "--instruction",
        type=str,
        choices=list(INSTRUCTIONS.keys()),
        default="no_instruct",
        help=f"The key for the instruction to use for embedding. Defaults to 'none'. Available keys: {', '.join(INSTRUCTIONS.keys())}"
    )
    parser.add_argument(
        "--translation",
        action="store_true",
        help="If flag set, the English translation of the questions will be embedded. Make sure it exists in the CSV file!"
    )
    args = parser.parse_args()
    instruction_key = args.instruction
    translation = bool(args.translation)
    instruction = INSTRUCTIONS[instruction_key]

    db_path = f"./data/query_embeddings/qwen3_query_embeddings{'_translation' if translation else ''}_{instruction_key}.db"
      
    print(f"Using instruction key: '{instruction_key}'")
    print(f"Instruction: {instruction}")
    print(f"Database path: {db_path}")
    print(f"Embedding {'translated English questions' if translation else 'original questions'}")

    create_db_and_table(db_path)
    
    print("Loading questions from CSV...")
    df = pd.read_csv(ECHR_QA_CSV_PATH)
    if translation:  
        queries = df[Column.QUESTION_TRANSLATION].tolist()
    else:
        queries = df[Column.QUESTION].tolist()
    
    existing_queries = get_existing_queries(db_path)
    
    queries_to_embed = [q for q in queries if q not in existing_queries]
    
    if not queries_to_embed:
        print("All queries are already embedded and stored for this instruction.")
        return
        
    print(f"Found {len(queries_to_embed)} new queries to embed.")

    print("Initializing Qwen embedding model...")
    embedding_function = QwenEmbeddings(retrieve_instruction=instruction)

    print("Embedding queries and storing them in the database...")
    batch_size = 2000
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for i in range(0, len(queries_to_embed), batch_size):
            batch_queries = queries_to_embed[i:i + batch_size]
            print(f"Processing batch {(i // batch_size) + 1}/{(len(queries_to_embed) + batch_size - 1) // batch_size}...")

            batch_embeddings = embedding_function.embed_query(batch_queries)
            
            data_to_insert = [
                (query, json.dumps(embedding))
                for query, embedding in zip(batch_queries, batch_embeddings)
            ]
            
            cursor.executemany(f"""
            INSERT OR REPLACE INTO {TABLE_NAME} (query, embedding)
            VALUES (?, ?)
            """, data_to_insert)
        conn.commit()

    print(f"Successfully embedded and stored all new queries for instruction '{instruction_key}'.")

if __name__ == "__main__":
    main()
