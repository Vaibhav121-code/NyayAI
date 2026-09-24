import hashlib
import logging
from typing import Any

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def _word_tokens(text: str) -> list[str]:
    return text.split()


def chunk_text(text: str, document_id: str, file_name: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict[str, Any]]:
    """Split text into deterministic chunks of approximately chunk_size words with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = _word_tokens(text)
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []

    for index in range(0, len(words), step):
        chunk_words = words[index:index + chunk_size]
        if not chunk_words:
            break

        chunk_text_value = " ".join(chunk_words)
        chunk_id = hashlib.sha256(f"{document_id}:{file_name}:{index}".encode("utf-8")).hexdigest()[:16]

        start_char = text.find(chunk_text_value)
        end_char = start_char + len(chunk_text_value) if start_char != -1 else -1

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_value,
            "start_position": start_char,
            "end_position": end_char,
            "metadata": {
                "document_id": document_id,
                "file_name": file_name,
                "chunk_id": chunk_id,
                "page_start": None,
                "page_end": None,
            },
        })

        if index + chunk_size >= len(words):
            break

    return chunks
