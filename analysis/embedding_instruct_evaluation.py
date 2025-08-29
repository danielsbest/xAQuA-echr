import os
import glob
import pandas as pd
from tqdm import tqdm
import sys
import json
import sqlite3
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.constants import ECHR_QA_CSV_PATH
from src.column import Column
from src.retrievers.qwen3 import get_db

def discover_chroma_dbs(base_path: str) -> dict[str, str]:
    """Discovers qwen3 Chroma DB directories in the given base path."""
    pattern = os.path.join(base_path, 'chroma_qwen3_db*')
    db_paths = glob.glob(pattern)
    db_map = {}
    for db_path in db_paths:
        db_name = os.path.basename(db_path)
        if db_name == 'chroma_qwen3_db':
            db_map['no_instruct'] = db_path
        else:
            instruct_name = db_name.replace('chroma_qwen3_db_instruct_', '')
            db_map[instruct_name] = db_path
    return db_map

def discover_query_embeddings(base_path: str) -> dict[str, str]:
    """Discovers qwen3 query embedding database files in the given base path."""
    pattern = os.path.join(base_path, 'qwen3_query_embeddings*.db')
    query_paths = glob.glob(pattern)
    query_map = {}
    prefix = 'qwen3_query_embeddings'
    for query_path in query_paths:
        filename = os.path.basename(query_path)
        name, _ = os.path.splitext(filename)

        if name == prefix:
            query_map['no_instruct'] = query_path
        elif name.startswith(prefix + '_'):
            instruct_name = name[len(prefix) + 1:]
            query_map[instruct_name] = query_path
    return query_map

def extract_case_ids(answer: str) -> list[dict[str, str]]:
    """
    Extracts case IDs from a JSON string answer.
    The input is expected to be a JSON array of objects, where each object
    can contain citations with multilingual case IDs.
    Returns a list of unique case ID groups, where each group is a dictionary
    mapping language code to case ID.
    """
    try:
        data = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return []

    unique_ids_set = set()

    if not isinstance(data, list):
        return []

    for item in data:
        if not isinstance(item, dict) or 'citations' not in item:
            continue
        
        citations = item['citations']
        if not isinstance(citations, list):
            continue

        for citation in citations:
            if not isinstance(citation, dict) or 'multilingual' not in citation:
                continue
            
            multilingual_data = citation['multilingual']
            if not isinstance(multilingual_data, dict):
                continue

            lang_id_map = {
                lang_code: lang_data['id']
                for lang_code, lang_data in multilingual_data.items()
                if isinstance(lang_data, dict) and 'id' in lang_data
            }
            
            if lang_id_map:
                unique_ids_set.add(frozenset(lang_id_map.items()))

    return [dict(t) for t in unique_ids_set]

def get_golden_retrievals(answer: str) -> list[str]:
    """Extracts golden case IDs from the answer field."""
    if pd.isna(answer):
        return []
    case_id_dicts = extract_case_ids(answer)
    golden_ids = [d['eng'] for d in case_id_dicts if 'eng' in d]
    return list(set(golden_ids))

def calculate_precision(retrieved: list[str], golden: list[str]) -> float:
    """Calculates precision."""
    retrieved_set = set(retrieved)
    golden_set = set(golden)
    if not retrieved_set:
        return 0.0
    true_positives = len(retrieved_set.intersection(golden_set))
    return true_positives / len(retrieved_set)

def calculate_recall(retrieved: list[str], golden: list[str]) -> float:
    """Calculates recall."""
    retrieved_set = set(retrieved)
    golden_set = set(golden)
    if not golden_set:
        return 1.0 if not retrieved_set else 0.0
    true_positives = len(retrieved_set.intersection(golden_set))
    return true_positives / len(golden_set)

def calculate_f1(precision: float, recall: float) -> float:
    """Calculates F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_mrr(retrieved: list[str], golden: list[str]) -> float:
    """Calculates Mean Reciprocal Rank."""
    golden_set = set(golden)
    for i, doc_id in enumerate(retrieved):
        if doc_id in golden_set:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg_at_k(retrieved: list[str], golden: list[str], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain (nDCG) at k."""
    retrieved_at_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_at_k):
        if doc_id in golden:
            dcg += 1.0 / np.log2(i + 2)

    ideal_dcg = 0.0
    num_golden = len(golden)
    for i in range(min(k, num_golden)):
        ideal_dcg += 1.0 / np.log2(i + 2)

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def run_evaluation():
    k_values = [10, 20]
    max_k = max(k_values)

    data_path = os.path.join(project_root, 'data')
    query_embeddings_path = os.path.join(data_path, 'query_embeddings') 
    chroma_dbs_path = os.path.join(data_path, 'chromadbs')

    doc_dbs = discover_chroma_dbs(chroma_dbs_path)
    query_embedding_files = discover_query_embeddings(query_embeddings_path)

    if not doc_dbs:
        print("No Chroma DBs found. Please check the 'data/' directory.")
        return
    
    if not query_embedding_files:
        print("No query embedding files found. Please check the 'data/query_embeddings/' directory.")
        return

    questions_df = pd.read_csv(ECHR_QA_CSV_PATH)

    results = []

    for db_name, doc_db_path in tqdm(doc_dbs.items(), desc="Document DBs"):
        print(f"\nLoading Document DB: {db_name}")
        db = get_db(db_path=doc_db_path)

        for query_embedding_name, query_embedding_path in tqdm(query_embedding_files.items(), desc="Query Embeddings", leave=False):
            with sqlite3.connect(query_embedding_path) as conn:
                query_embeddings_df = pd.read_sql_query("SELECT query, embedding FROM query_embeddings", conn)
            
            query_embeddings_df['embedding'] = query_embeddings_df['embedding'].apply(json.loads)
            query_embeddings_df = query_embeddings_df.set_index('query')

            for _, row in tqdm(questions_df.iterrows(), desc="Questions", leave=False, total=len(questions_df)):
                if 'translation' in query_embedding_name:
                    question = row[Column.QUESTION_TRANSLATION]
                else:
                    question = row[Column.QUESTION]
                guide_id = row['guide_id']
                group_id = row['group_id']
                question_id = f"{guide_id}_{group_id}"
        
                query_embedding:list[float] = query_embeddings_df.loc[question, 'embedding']

        
                retrieved_docs_with_scores = db.similarity_search_by_vector_with_relevance_scores(query_embedding, k=max_k)
                all_retrieved_doc_ids = [doc.metadata['case_id'] for doc, _ in retrieved_docs_with_scores]
        
                golden_doc_ids = get_golden_retrievals(row['answer'])

                result_row = {
                    'question_id': question_id,
                    'doc_db': db_name,
                    'query_embedding': query_embedding_name,
                    'golden_ids': golden_doc_ids
                }

                for k in k_values:
                    retrieved_doc_ids_at_k = all_retrieved_doc_ids[:k]
                    
                    precision = calculate_precision(retrieved_doc_ids_at_k, golden_doc_ids)
                    recall = calculate_recall(retrieved_doc_ids_at_k, golden_doc_ids)
                    f1 = calculate_f1(precision, recall)
                    mrr = calculate_mrr(retrieved_doc_ids_at_k, golden_doc_ids)
                    ndcg = calculate_ndcg_at_k(retrieved_doc_ids_at_k, golden_doc_ids, k)
                    
                    result_row[f'precision@{k}'] = precision
                    result_row[f'recall@{k}'] = recall
                    result_row[f'f1@{k}'] = f1
                    result_row[f'mrr@{k}'] = mrr
                    result_row[f'ndcg@{k}'] = ndcg
                    result_row[f'retrieved_ids@{k}'] = retrieved_doc_ids_at_k

                results.append(result_row)
    
    if not results:
        print("\nEvaluation loop was skipped as no query embeddings were found.")
        print("Please generate query embeddings and place them in 'data/query_embeddings/'.")
        return

    results_df = pd.DataFrame(results)
    print("\n--- Individual Results ---")
    print(results_df.head())
    results_df.to_csv(os.path.join(data_path, "analysis", "embedding_eval_results.csv"), index=False)

    agg_dict = {}
    for k in k_values:
        agg_dict[f'precision@{k}'] = 'mean'
        agg_dict[f'recall@{k}'] = 'mean'
        agg_dict[f'f1@{k}'] = 'mean'
        agg_dict[f'mrr@{k}'] = 'mean'
        agg_dict[f'ndcg@{k}'] = 'mean'

    aggregated_results = results_df.groupby(['doc_db', 'query_embedding']).agg(agg_dict).reset_index()

    print("\n--- Aggregated Results ---")
    print(aggregated_results)
    aggregated_results.to_csv(os.path.join(data_path, "analysis", "embedding_eval_aggregated_results.csv"), index=False)


if __name__ == '__main__':
    run_evaluation()
