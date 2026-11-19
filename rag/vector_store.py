
import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from rag.config import CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)


def initialize_collection(
    collection_name: str = COLLECTION_NAME,
    chroma_dir: str | Path = CHROMA_DIR,
) -> Any:
    """Initialize a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(name=collection_name)


def add_documents(
    collection: Any,
    documents: list[str],
    embeddings: np.ndarray,
    metadata_list: list[dict[str, Any]],
) -> None:
    """Insert or update chunks using their deterministic chunk IDs."""
    if len(documents) != len(metadata_list) or len(documents) != len(embeddings):
        raise ValueError(
            "documents, embeddings, and metadata_list must have equal lengths"
        )

    if not documents:
        return

    ids: list[str] = []

    for metadata in metadata_list:
        chunk_id = metadata.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Every chunk must have a non-empty chunk_id")

        ids.append(chunk_id)

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_id values found in this batch")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadata_list,
    )

    logger.info("Upserted %d chunks into ChromaDB", len(documents))


def search(
    collection: Any,
    query_embedding: np.ndarray,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Query the collection with an embedding vector."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_options = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": top_k,
        "include": ["documents", "distances", "metadatas"],
    }
    if where is not None:
        query_options["where"] = where
    return collection.query(**query_options)


def count_documents(collection: Any) -> int:
    """Return the total number of stored chunks."""
    return collection.count()
