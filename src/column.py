from enum import Enum


class Column(str, Enum):
    GENERATED_ANSWER = "generated_answer"
    GENERATED_CITATIONS = "generated_citations"
    TARGET_CITATIONS = "citations"
    QUESTION = "question"
    QUESTION_TRANSLATION = "question_eng"
    TARGET_ANSWER = "answer"
    CITATIONS = "citations"
    CORRECTNESS_BERT = "correctness_bert"
    CORRECTNESS_ROUGE = "correctness_rouge"
    CLAIM_RECALL = "claim_recall"
    LLM_CLAIM_RECALL = "llm_claim_recall"
    LLM_FAITHFULNESS = "llm_faithfulness"
    LLM_CITATION_CORRECTNESS = "llm_citation_correctness"
    LLM_CITATION_SEMANTIC_SIMILARITY = "llm_citation_semantic_similarity"
    CITATION_FAITHFULNESS = "citation_faithfulness"
    CITATION_SIMILARITY_EM = "citation_similarity_em"
    CITATION_SIMILARITY_NLI = "citation_similarity_nli"

    # echr_case_paragraphs.csv
    ECHR_CASE_ID = "case_id"
    ECHR_CASE_NAME = "case_name"
    ECHR_CASE_PARAGRAPH_NUM = "paragraph_number"
    ECHR_CASE_PARAGRAPH_TEXT = "paragraph_text"

