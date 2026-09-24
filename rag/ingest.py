import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from rag.cleaner import clean_text
from rag.chunker import chunk_text
from rag.config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, TOP_K
from rag.embeddings import embed_texts
from rag.pdf_loader import extract_text
from rag.vector_store import add_documents, count_documents, initialize_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def stable_document_id(pdf_path: Path) -> str:
    """Generate a deterministic document ID from the source path."""
    return hashlib.sha256(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:32]


def discover_pdfs(data_dir: str | Path = DATA_DIR) -> list[Path]:
    """Find all PDFs under the data/raw directory."""
    directory = Path(data_dir)
    if not directory.exists():
        logger.warning("Data directory does not exist: %s", directory)
        return []
    return sorted(p for p in directory.rglob("*.pdf") if p.is_file())


def process_pdf(pdf_path: Path) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert one PDF into cleaned chunks with metadata."""
    document_id = stable_document_id(pdf_path)
    document_data = extract_text(pdf_path)

    if not document_data.get("text") or not document_data.get("extractable"):
        logger.warning("No extractable text found in PDF: %s", pdf_path)
        return [], [], []

    cleaned = clean_text(document_data["text"])
    if not cleaned:
        logger.warning("Cleaned text is empty for PDF: %s", pdf_path)
        return [], [], []

    page_map = document_data.get("pages", [])
    chunks = chunk_text(cleaned, document_id=document_id, file_name=pdf_path.name)

    chunk_records: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = chunk["metadata"].copy()
        page_start = None
        page_end = None
        if page_map:
            estimated_start = max(1, min(len(page_map), max(1, int((metadata.get("start_position", 0) / max(len(cleaned), 1)) * len(page_map)))))
            page_start = estimated_start
            page_end = estimated_start
        metadata.update({
            "document_id": document_id,
            "file_name": pdf_path.name,
            "chunk_id": metadata["chunk_id"],
            "page_start": page_start,
            "page_end": page_end,
        })
        chunk_records.append(metadata)

    return [chunk["text"] for chunk in chunks], chunks, chunk_records


def ingest_documents(data_dir: str | Path = DATA_DIR) -> dict[str, int]:
    """Ingest all PDFs found in the raw data directory into ChromaDB."""
    pdf_files = discover_pdfs(data_dir)
    if not pdf_files:
        logger.warning("No PDFs found in %s. Add judgment PDFs to begin ingestion.", data_dir)
        return {"documents_processed": 0, "chunks_created": 0, "chunks_stored": 0}

    collection = initialize_collection(collection_name=COLLECTION_NAME, chroma_dir=CHROMA_DIR)
    total_chunks = 0

    logger.info("Processing documents...")
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            chunk_texts, chunk_data, metadata_list = process_pdf(pdf_path)
            if not chunk_texts:
                continue

            embeddings = embed_texts(chunk_texts)
            if embeddings.size == 0:
                logger.warning("No embeddings created for %s", pdf_path)
                continue

            add_documents(collection, chunk_texts, embeddings, metadata_list)
            total_chunks += len(chunk_texts)
            logger.info("Stored %d chunks from %s", len(chunk_texts), pdf_path.name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to ingest %s: %s", pdf_path, exc)
            continue

    stored_count = count_documents(collection)
    return {
        "documents_processed": len(pdf_files),
        "chunks_created": total_chunks,
        "chunks_stored": stored_count,
    }


def main() -> None:
    result = ingest_documents()
    print("Ingestion completed.")
    print(f"Documents processed: {result['documents_processed']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Chunks stored: {result['chunks_stored']}")


if __name__ == "__main__":
    main()
