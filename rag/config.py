import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _resolve_path(env_name: str, default: Path) -> Path:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate

    candidate_from_root = (BASE_DIR / candidate).resolve()
    if candidate_from_root.exists():
        return candidate_from_root

    if raw_value.startswith("../"):
        normalized = candidate.as_posix().lstrip("../")
        fallback = (BASE_DIR / normalized).resolve()
        if fallback.exists() or not normalized.startswith("/"):
            return fallback

    return candidate_from_root


DATA_DIR = _resolve_path("DATA_DIR", BASE_DIR / "data" / "raw")
CHROMA_DIR = _resolve_path("CHROMA_DIR", BASE_DIR / "data" / "chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nyaai_judgments_v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Upgraded ingestion configuration
INGESTION_COLLECTION_NAME = os.getenv(
    "INGESTION_COLLECTION_NAME",
    "nyaai_judgments_v2",
)

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "500"))

TRACKING_DB = BASE_DIR / "data" / "ingestion_state.sqlite3"
FAILED_LOG = BASE_DIR / "data" / "ingestion_failures.csv"
__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "CHROMA_DIR",
    "COLLECTION_NAME",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "EMBEDDING_MODEL",
    "INGESTION_COLLECTION_NAME",
    "EMBEDDING_BATCH_SIZE",
    "CHROMA_BATCH_SIZE",
    "TRACKING_DB",
    "FAILED_LOG",
]
