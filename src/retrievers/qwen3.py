from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import torch
from typing import Optional
import sqlite3
import json
import os

_model_instance: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    """
    Singleton getter for the Qwen3 embedding model.
    On first call, loads the model (to GPU, half-precision, flash attention).
    Subsequent calls return the exact same instance.
    """
    global _model_instance
    if _model_instance is None:
        model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-4B",
            model_kwargs={"attn_implementation": "flash_attention_2"},
            tokenizer_kwargs={"padding_side": "left"},
            cache_folder="cache/",
            device="cuda",
        )
        model = model.half()
        _model_instance = model
    return _model_instance

def get_doc_instruct(task_description: str, documents: list[str]) -> list[str]:
    instruct_documents = [f'Instruct: {task_description}\nDocument:{document}' for document in documents]
    return instruct_documents

def get_query_instruct(task_description: str, query: list[str] | str) -> list[str]:
    if isinstance(query, str):
        query = [query]
    queries = [f'Instruct: {task_description}\nQuery:{query}' for query in query]
    return queries


class QwenEmbeddings:

    def __init__(self, embed_instruction: Optional[str] = None, retrieve_instruction: Optional[str] = None) -> None:
        self.embed_instruction = embed_instruction
        self.retrieve_instruction = retrieve_instruction
        self.model = get_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.embed_instruction:
            texts = get_doc_instruct(task_description=self.embed_instruction, documents=texts)
        embeddings = self.model.encode(texts, convert_to_numpy=True, dtype=torch.float16)
        embeddings = embeddings.tolist()
        return embeddings

    def embed_query(self, texts: list[str]|str) -> list[float] | list[list[float]]:
        is_str = isinstance(texts, str)
        if self.retrieve_instruction:
            texts = get_query_instruct(task_description=self.retrieve_instruction, query=texts)
        embeddings = self.model.encode(texts, convert_to_numpy=True, dtype=torch.float16)
        embeddings = embeddings.tolist()
        if is_str:
            return embeddings[0]
        else:
            return embeddings


def get_db(db_path: str, embedding_function=None) -> Chroma:
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function,
    )
    print("DB Loaded - Entries:", db._collection.count())
    return db


class QwenRetriever:
    def __init__(self, query_db_path="data/query_embeddings/qwen3_query_embeddings.db", chroma_db_path="./data/chromadbs/chroma_qwen3_db") -> None:
        self.db = get_db(db_path=chroma_db_path)
        self.query_db_path = query_db_path
        self._check_query_db()

    def _check_query_db(self):
        if not os.path.exists(self.query_db_path):
            raise FileNotFoundError(f"Query database not found at {self.query_db_path}. Please run setup/preembed_qwen3_queries.py first.")

    def _get_embedding_from_db(self, query: str) -> Optional[list[float]]:
        with sqlite3.connect(self.query_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM query_embeddings WHERE query = ?", (query,))
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
        return None

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        embedding = self._get_embedding_from_db(query)
        if embedding:
            return self.db.similarity_search_by_vector_with_relevance_scores(embedding, k=k)
        else:
            print(f"Query '{query}' not found in the pre-embedded query database.")
            return []

