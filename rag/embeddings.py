import logging
import os
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
_MODEL: SentenceTransformer | None = None


def load_embedding_model() -> SentenceTransformer:
    """Load the configured embedding model on CUDA when available, else CPU."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        device = "cpu"

    logger.info("Loading embedding model '%s' on %s", _DEFAULT_MODEL, device)
    _MODEL = SentenceTransformer(_DEFAULT_MODEL, device=device)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed multiple texts in batches."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = load_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([query])[0]
