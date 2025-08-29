import os
from pydantic import SecretStr

from dotenv import load_dotenv
from pathlib import Path

from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
data_dir = project_root / "data" / "embedding_langchain_cache"

fs_store = LocalFileStore(str(data_dir))


def get_cluster_embedder():
    cluster_embedder_model = GoogleGenerativeAIEmbeddings(
        google_api_key=SecretStr(os.getenv("GOOGLE_API_KEY")),
        model="models/gemini-embedding-exp-03-07", 
        task_type="CLUSTERING",
    )

    cluster_embedder = CacheBackedEmbeddings.from_bytes_store(
    cluster_embedder_model,
    document_embedding_cache=fs_store,
    batch_size=10,
    query_embedding_cache=True,
    namespace=f"{cluster_embedder_model.model}_{cluster_embedder_model.task_type}",
    )


    return cluster_embedder


def get_semantic_embedder():
    semantic_embedder_model = GoogleGenerativeAIEmbeddings(
        google_api_key=SecretStr(os.getenv("GOOGLE_API_KEY")),
        model="models/gemini-embedding-exp-03-07", 
        task_type="SEMANTIC_SIMILARITY",
    )

    semantic_embedder = CacheBackedEmbeddings.from_bytes_store(
        semantic_embedder_model,
        document_embedding_cache=fs_store,
        query_embedding_cache=True,
        namespace=f"{semantic_embedder_model.model}_{semantic_embedder_model.task_type}",
    )
    
    return semantic_embedder