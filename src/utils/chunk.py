from typing import TypeVar, List

T = TypeVar("T")


def chunk(tokens: list[T], max_length: int, stride: int) -> list[list[T]]:
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i : i + max_length]
        chunks.append(chunk)

        if i + max_length >= len(tokens):
            break
    return chunks
