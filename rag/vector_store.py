import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from rag.config import CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)


def initialize_collection(collection_name: str = COLLECTION_NAME, chroma_dir: str | Path = CHROMA_DIR) -> Any:
    """Initialize a persistent ChromaDB collection without recreating it on every run."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def _stable_id(document_id: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{document_id}:{chunk_index}".encode("utf-8")).hexdigest()


def add_documents(collection: Any, documents: list[str], embeddings: np.ndarray, metadata_list: list[dict[str, Any]]) -> None:
    """Add document chunks and metadata to the collection with unique IDs."""
    ids: list[str] = []
    for idx, metadata in enumerate(metadata_list):
        document_id = metadata.get("document_id", "unknown")
        ids.append(_stable_id(document_id, idx))

    existing = collection.get(ids=ids, include=[])
    existing_ids = set(existing.get("ids", []))

    filtered_ids: list[str] = []
    filtered_docs: list[str] = []
    filtered_embeddings: list[list[float]] = []
    filtered_metadata: list[dict[str, Any]] = []

    for idx, doc_id in enumerate(ids):
        if doc_id in existing_ids:
            continue
        filtered_ids.append(doc_id)
        filtered_docs.append(documents[idx])
        filtered_embeddings.append(embeddings[idx].tolist())
        filtered_metadata.append(metadata_list[idx])

    if filtered_docs:
        collection.add(
            ids=filtered_ids,
            documents=filtered_docs,
            embeddings=filtered_embeddings,
            metadatas=filtered_metadata,
        )


def search(collection: Any, query_embedding: np.ndarray, top_k: int = 5) -> dict[str, Any]:
    """Query the collection with an embedding vector and return top-k matches."""
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )


def count_documents(collection: Any) -> int:
    """Return the total number of stored documents."""
    return collection.count()
