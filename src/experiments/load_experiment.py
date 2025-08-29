from enum import Enum
import os
import csv

import pandas as pd

from src.constants import ECHR_QA_CSV_PATH


# python eval.py --experiment base_llama_8b --device cuda:0 --metric "correctness_bert, correctness_rouge, claim_recall, citation_faithfulness, citation_similarity_em, citation_similarity_nli"
class Experiment(str, Enum):
    # Base Experiments
    RON_BASE_R10528_NOTRANSLATIION = "ron_base_r10528_notranslation"
    RON_BASE_R10528_FULLTRANSLATION = "ron_base_r10528_fulltranslation"
    RON_BASE_R10528_HALFTRANSLATION = "ron_base_r10528_halftranslation"


    # RAG Experiments

    ## R1_0528 LLM
    # Baseline
    RON_RAG_QWEN3_DOC_NO_INSTRUCT_QUERY_BASE_R10528_NOTRANSLATION = "ron_rag_qwen3_doc_no_instruct_query_base_r10528_notranslation"
    # Best performer
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_R10528_NOTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_echr_retrieve_r10528_notranslation"
    # Query instruct in english revealing romanian
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_RON_R10528_NOTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_echr_retrieve_ron_r10528_notranslation"
    # Query instruct in romanian
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_ECHR_RETRIEVE_RON_R10528_NOTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_ron_echr_retrieve_ron_r10528_notranslation"
    # Translation variants
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_R10528_HALFTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_translation_echr_retrieve_r10528_halftranslation"
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_R10528_FULLTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_translation_echr_retrieve_r10528_fulltranslation"
    # Fullnative variant
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_R10528_FULLNATIVE = "ron_rag_qwen3_doc_echr_topic_query_echr_retrieve_r10528_fullnative"

    ## GPT-OSS-120B LLM
    # Baseline
    RON_RAG_QWEN3_DOC_NO_INSTRUCT_QUERY_BASE_GPTOSS_NOTRANSLATION = "ron_rag_qwen3_doc_no_instruct_query_base_gptoss_notranslation"
    # Best performer
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_GPTOSS_NOTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_echr_retrieve_gptoss_notranslation"
    # Query instruct in english revealing romanian
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_RON_GPTOSS_NOTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_echr_retrieve_ron_gptoss_notranslation"
    # Query instruct in romanian
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_ECHR_RETRIEVE_RON_GPTOSS_NOTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_ron_echr_retrieve_ron_gptoss_notranslation"
    # Translation variants
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_GPTOSS_HALFTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_translation_echr_retrieve_gptoss_halftranslation"
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_GPTOSS_FULLTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_translation_echr_retrieve_gptoss_fulltranslation"
    # Fullnative variant
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_GPTOSS_FULLNATIVE = "ron_rag_qwen3_doc_echr_topic_query_echr_retrieve_gptoss_fullnative"
 
    ## K2 LLM
    # Baseline
    RON_RAG_QWEN3_DOC_NO_INSTRUCT_QUERY_BASE_K2_NOTRANSLATION = "ron_rag_qwen3_doc_no_instruct_query_base_k2_notranslation"
    # Best performer
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_K2_NOTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_echr_retrieve_k2_notranslation"
    # Query instruct in english revealing romanian
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_RON_K2_NOTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_echr_retrieve_ron_k2_notranslation"
    # Query instruct in romanian
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_ECHR_RETRIEVE_RON_K2_NOTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_ron_echr_retrieve_ron_k2_notranslation"
    # Translation variants
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_K2_HALFTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_translation_echr_retrieve_k2_halftranslation"
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_K2_FULLTRANSLATION = "ron_rag_qwen3_doc_echr_topic_query_translation_echr_retrieve_k2_fulltranslation"
    # Fullnative variant
    RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_K2_FULLNATIVE = "ron_rag_qwen3_doc_echr_topic_query_echr_retrieve_k2_fullnative"


def load_experiment_df(e: Experiment):
    path = "data/e_" + e + ".csv"
    return (
        pd.read_csv(path) if os.path.exists(path) else pd.read_csv(ECHR_QA_CSV_PATH)
    ), path