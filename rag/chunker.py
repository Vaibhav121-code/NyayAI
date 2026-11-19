
import hashlib
import re
from typing import Any

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_pages(
    pages: list[dict[str, Any]],
    document_id: str,
    file_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Create chunks while preserving exact source page numbers."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []

    for page in pages:
        page_number = int(page["page_number"])
        text = page.get("text", "").strip()

        if not text:
            continue

        tokens = list(re.finditer(r"\S+", text))

        for chunk_index, start in enumerate(range(0, len(tokens), step)):
            selected = tokens[start:start + chunk_size]

            if not selected:
                break

            start_char = selected[0].start()
            end_char = selected[-1].end()
            chunk_value = text[start_char:end_char]

            chunk_id = hashlib.sha256(
                f"{document_id}:{page_number}:{chunk_index}".encode("utf-8")
            ).hexdigest()[:32]

            metadata = {
                "document_id": document_id,
                "file_name": file_name,
                "chunk_id": chunk_id,
                "page_start": page_number,
                "page_end": page_number,
                "start_position": start_char,
                "end_position": end_char,
            }

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_value,
                "start_position": start_char,
                "end_position": end_char,
                "metadata": metadata,
            })

            if start + chunk_size >= len(tokens):
                break

    return chunks


def chunk_text(
    text: str,
    document_id: str,
    file_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Backward-compatible text chunking function."""

    return chunk_pages(
        [{"page_number": 1, "text": text}],
        document_id,
        file_name,
        chunk_size,
        overlap,
    )
