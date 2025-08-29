from typing import Protocol
from langchain_core.documents import Document


class Retriever(Protocol):

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Document, float]]: ...
