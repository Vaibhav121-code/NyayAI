from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from tqdm import tqdm

from rag.chunker import chunk_pages
from rag.cleaner import clean_text
from rag.config import (
    CHROMA_BATCH_SIZE,
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_BATCH_SIZE,
    FAILED_LOG,
    INGESTION_COLLECTION_NAME,
    TRACKING_DB,
)
from rag.embeddings import embed_texts
from rag.ingestion_tracker import IngestionTracker
from rag.pdf_loader import extract_text
from rag.vector_store import add_documents, count_documents, initialize_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CanonicalSources:
    groups: list[list[Path]]
    duplicate_groups: list[list[Path]]
    hashes: dict[Path, str]
    hash_errors: dict[Path, str]


@dataclass
class IngestionPlan:
    discovered_count: int
    sources: CanonicalSources
    completed: list[Path]
    completed_files: list[Path]
    pending: list[Path]
    selected: list[Path]


def _relative_pdf_path(pdf_path: Path, data_dir: str | Path) -> str:
    return pdf_path.resolve().relative_to(Path(data_dir).resolve()).as_posix()


def stable_document_id(pdf_path: Path, data_dir: str | Path = DATA_DIR) -> str:
    """Generate a deterministic document ID from its path under the data root."""
    relative_path = _relative_pdf_path(pdf_path, data_dir)
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:32]


def calculate_content_hash(pdf_path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Calculate a file's SHA-256 digest without loading it into memory."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_pdfs(data_dir: str | Path = DATA_DIR) -> list[Path]:
    """Find PDFs recursively and sort them by normalized relative path."""
    directory = Path(data_dir)
    if not directory.exists():
        logger.warning("Data directory does not exist: %s", directory)
        return []

    pdfs = (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    return sorted(pdfs, key=lambda path: path.relative_to(directory).as_posix().casefold())


def _hash_candidate(pdf_path: Path) -> tuple[Path, str | None, str | None]:
    try:
        return pdf_path, calculate_content_hash(pdf_path), None
    except Exception as exc:  # noqa: BLE001
        return pdf_path, None, str(exc)


def select_canonical_sources(
    pdf_files: list[Path],
    data_dir: str | Path = DATA_DIR,
) -> CanonicalSources:
    """Choose one deterministic path for each byte-identical PDF source."""
    root = Path(data_dir).resolve()
    by_size: dict[int, list[Path]] = {}
    for pdf_path in pdf_files:
        by_size.setdefault(pdf_path.stat().st_size, []).append(pdf_path)

    hash_candidates = [path for same_size_paths in by_size.values() if len(same_size_paths) > 1 for path in same_size_paths]
    hashes: dict[Path, str] = {}
    hash_errors: dict[Path, str] = {}
    if hash_candidates:
        with ThreadPoolExecutor(max_workers=min(8, len(hash_candidates))) as executor:
            for path, digest, error in executor.map(_hash_candidate, hash_candidates):
                if digest is not None:
                    hashes[path] = digest
                elif error is not None:
                    hash_errors[path] = error

    paths_by_hash: dict[str, list[Path]] = {}
    for path, digest in hashes.items():
        paths_by_hash.setdefault(digest, []).append(path)

    def path_key(path: Path) -> str:
        return path.relative_to(root).as_posix().casefold()

    duplicate_groups = [
        sorted(paths, key=path_key)
        for paths in paths_by_hash.values()
        if len(paths) > 1
    ]
    duplicate_groups.sort(key=lambda group: path_key(group[0]))
    duplicate_paths = {path for group in duplicate_groups for path in group}
    groups = duplicate_groups + [[path] for path in pdf_files if path not in duplicate_paths]
    groups.sort(key=lambda group: path_key(group[0]))
    return CanonicalSources(groups, duplicate_groups, hashes, hash_errors)


def process_pdf(
    pdf_path: Path,
    data_dir: str | Path = DATA_DIR,
    content_hash: str | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract and clean each page, then create versioned chunks with source metadata."""
    digest = content_hash or calculate_content_hash(pdf_path)
    document_id = stable_document_id(pdf_path, data_dir)
    document_data = extract_text(pdf_path)

    if document_data.get("error"):
        raise RuntimeError(f"PDF extraction failed: {document_data['error']}")
    if not document_data.get("extractable"):
        raise ValueError("PDF has no extractable text; OCR was not attempted")

    cleaned_pages: list[dict[str, Any]] = []
    for page in document_data.get("pages", []):
        text = clean_text(page.get("text", ""))
        if text:
            cleaned_pages.append({"page_number": int(page["page_number"]), "text": text})

    if not cleaned_pages:
        raise ValueError("PDF has no text after cleaning; OCR was not attempted")

    chunks = chunk_pages(
        cleaned_pages,
        document_id=document_id,
        file_name=pdf_path.name,
    )
    if not chunks:
        raise ValueError("PDF produced no chunks")

    for chunk in chunks:
        metadata = chunk["metadata"].copy()
        base_chunk_id = metadata["chunk_id"]
        versioned_chunk_id = hashlib.sha256(
            f"{base_chunk_id}:{digest}".encode("utf-8")
        ).hexdigest()[:32]
        metadata.update({
            "chunk_id": versioned_chunk_id,
            "content_hash": digest,
        })
        chunk["chunk_id"] = versioned_chunk_id
        chunk["metadata"] = metadata

    chunk_texts = [chunk["text"] for chunk in chunks]
    metadata_list = [chunk["metadata"] for chunk in chunks]
    return chunk_texts, chunks, metadata_list


def _document_index(collection: Any, document_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    result = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )
    return result.get("ids", []), result.get("metadatas", [])


def _indexed_data_is_consistent(
    collection: Any,
    document_id: str,
    content_hash: str,
    expected_count: int,
) -> bool:
    ids, metadatas = _document_index(collection, document_id)
    return (
        len(ids) == expected_count
        and len(metadatas) == expected_count
        and all(metadata and metadata.get("content_hash") == content_hash for metadata in metadatas)
    )


def _open_existing_collection(collection_name: str, chroma_dir: str | Path) -> Any | None:
    """Open an existing collection without creating it, for dry-run inspection."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        return client.get_collection(name=collection_name)
    except Exception as exc:  # noqa: BLE001
        if "does not exist" in str(exc).casefold():
            return None
        raise


def build_ingestion_plan(
    pdf_files: list[Path],
    data_dir: str | Path,
    limit: int | None,
    resume: bool,
    tracker_records: dict[str, dict[str, Any]] | None = None,
    collection: Any | None = None,
    sources: CanonicalSources | None = None,
) -> IngestionPlan:
    """Filter verified completed sources before applying the pending-document limit."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    root = Path(data_dir).resolve()
    canonical_sources = sources or select_canonical_sources(pdf_files, root)
    completed: list[Path] = []
    completed_files: list[Path] = []
    pending: list[Path] = []
    records = tracker_records or {}

    for group in canonical_sources.groups:
        canonical_path = group[0]
        completed_group_files: list[Path] = []
        if resume and collection is not None:
            for source_path in group:
                relative_path = _relative_pdf_path(source_path, root)
                record = records.get(relative_path)
                if record is None or record["status"] != "completed":
                    continue

                try:
                    content_hash = canonical_sources.hashes.get(source_path)
                    if content_hash is None:
                        content_hash = calculate_content_hash(source_path)
                    if (
                        record["content_hash"] == content_hash
                        and _indexed_data_is_consistent(
                            collection,
                            stable_document_id(source_path, root),
                            content_hash,
                            record["chunk_count"],
                        )
                    ):
                        completed_group_files.append(source_path)
                except OSError as exc:
                    canonical_sources.hash_errors[source_path] = str(exc)

        if completed_group_files:
            completed.append(canonical_path)
            completed_files.extend(completed_group_files)
        else:
            pending.append(canonical_path)

    eligible = pending if resume else [group[0] for group in canonical_sources.groups]
    selected = eligible if limit is None else eligible[:limit]
    return IngestionPlan(
        len(pdf_files),
        canonical_sources,
        completed,
        completed_files,
        pending,
        selected,
    )


def _plan_summary(plan: IngestionPlan) -> dict[str, int]:
    duplicate_files = sum(len(group) - 1 for group in plan.sources.duplicate_groups)
    return {
        "documents_discovered": plan.discovered_count,
        "exact_duplicate_groups": len(plan.sources.duplicate_groups),
        "duplicate_source_files": duplicate_files,
        "canonical_pdfs_selected": len(plan.sources.groups),
        "already_completed_pdfs": len(plan.completed_files),
        "already_completed_canonical_sources": len(plan.completed),
        "remaining_pdfs": len(plan.pending),
        "would_process_next_batch": len(plan.selected),
        "hash_errors": len(plan.sources.hash_errors),
    }


def _remove_stale_chunks(
    collection: Any,
    document_id: str,
    current_ids: set[str],
    batch_size: int,
) -> None:
    existing_ids, _ = _document_index(collection, document_id)
    stale_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in current_ids]
    for offset in range(0, len(stale_ids), batch_size):
        collection.delete(ids=stale_ids[offset:offset + batch_size])


def _append_failure(
    failed_log: str | Path,
    pdf_path: str,
    content_hash: str,
    error: str,
) -> None:
    log_path = Path(failed_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if needs_header:
            writer.writerow(["pdf_path", "content_hash", "error", "timestamp"])
        writer.writerow([
            pdf_path,
            content_hash,
            error,
            datetime.now(timezone.utc).isoformat(),
        ])


def ingest_documents(
    data_dir: str | Path = DATA_DIR,
    limit: int | None = None,
    resume: bool = False,
    collection_name: str = INGESTION_COLLECTION_NAME,
    tracking_db: str | Path = TRACKING_DB,
    failed_log: str | Path = FAILED_LOG,
    chroma_dir: str | Path = CHROMA_DIR,
    dry_run: bool = False,
) -> dict[str, int]:
    """Ingest canonical PDFs or return a read-only plan without writing data."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if EMBEDDING_BATCH_SIZE <= 0:
        raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
    if CHROMA_BATCH_SIZE <= 0:
        raise ValueError("CHROMA_BATCH_SIZE must be positive")

    root = Path(data_dir).resolve()
    discovered_pdfs = discover_pdfs(root)
    sources = select_canonical_sources(discovered_pdfs, root)

    if dry_run:
        collection = _open_existing_collection(collection_name, chroma_dir)
        plan = build_ingestion_plan(
            discovered_pdfs,
            root,
            limit,
            resume=True,
            tracker_records=IngestionTracker.read_all(tracking_db),
            collection=collection,
            sources=sources,
        )
        return _plan_summary(plan)

    collection = None
    tracker_records = None
    if resume:
        collection = initialize_collection(collection_name=collection_name, chroma_dir=chroma_dir)
        tracker_records = IngestionTracker.read_all(tracking_db)

    plan = build_ingestion_plan(
        discovered_pdfs,
        root,
        limit,
        resume=resume,
        tracker_records=tracker_records,
        collection=collection,
        sources=sources,
    )
    summary = _plan_summary(plan)
    summary.update({
        "documents_selected": len(plan.selected),
        "documents_skipped": len(plan.completed),
        "documents_processed": 0,
        "documents_failed": 0,
        "chunks_created": 0,
        "chunks_stored": 0,
    })

    if not plan.selected:
        if collection is not None:
            summary["chunks_stored"] = count_documents(collection)
        return summary

    if collection is None:
        collection = initialize_collection(collection_name=collection_name, chroma_dir=chroma_dir)
    tracker = IngestionTracker(tracking_db)

    logger.info("Processing %d canonical PDFs...", len(plan.selected))
    for pdf_path in tqdm(plan.selected, desc="Processing PDFs"):
        relative_path = _relative_pdf_path(pdf_path, root)
        content_hash = ""
        document_id = stable_document_id(pdf_path, root)

        try:
            content_hash = sources.hashes.get(pdf_path) or calculate_content_hash(pdf_path)

            tracker.mark_processing(relative_path, content_hash)
            chunk_texts, chunks, metadata_list = process_pdf(
                pdf_path,
                data_dir=root,
                content_hash=content_hash,
            )

            for offset in range(0, len(chunk_texts), CHROMA_BATCH_SIZE):
                batch_end = min(offset + CHROMA_BATCH_SIZE, len(chunk_texts))
                batch_texts = chunk_texts[offset:batch_end]
                batch_metadata = metadata_list[offset:batch_end]
                embedding_batches = []
                for embedding_offset in range(offset, batch_end, EMBEDDING_BATCH_SIZE):
                    embedding_end = min(embedding_offset + EMBEDDING_BATCH_SIZE, batch_end)
                    embeddings = embed_texts(chunk_texts[embedding_offset:embedding_end])
                    if embeddings.ndim != 2 or len(embeddings) != embedding_end - embedding_offset:
                        raise ValueError("Embedding output count does not match input text count")
                    embedding_batches.append(embeddings)

                batch_embeddings = np.vstack(embedding_batches).astype(np.float32, copy=False)
                add_documents(collection, batch_texts, batch_embeddings, batch_metadata)

            current_ids = {metadata["chunk_id"] for metadata in metadata_list}
            _remove_stale_chunks(collection, document_id, current_ids, CHROMA_BATCH_SIZE)
            tracker.mark_completed(relative_path, content_hash, len(chunk_texts))
            summary["documents_processed"] += 1
            summary["chunks_created"] += len(chunks)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.exception("Failed to ingest %s: %s", pdf_path, error)
            try:
                tracker.mark_failed(relative_path, content_hash, error)
            except Exception:  # noqa: BLE001
                logger.exception("Could not update tracking state for %s", pdf_path)
            try:
                _append_failure(failed_log, relative_path, content_hash, error)
            except Exception:  # noqa: BLE001
                logger.exception("Could not write failure report for %s", pdf_path)
            summary["documents_failed"] += 1

    summary["chunks_stored"] = count_documents(collection)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest judgment PDFs into ChromaDB.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of PDFs to process; with --resume, counts pending canonical PDFs",
    )
    parser.add_argument("--resume", action="store_true", help="Skip only completed, verified PDFs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the next resumable batch without modifying the tracker or ChromaDB",
    )
    parser.add_argument(
        "--collection",
        default=INGESTION_COLLECTION_NAME,
        help=f"Chroma collection to write (default: {INGESTION_COLLECTION_NAME})",
    )
    args = parser.parse_args()

    result = ingest_documents(
        limit=args.limit,
        resume=args.resume or args.dry_run,
        collection_name=args.collection,
        dry_run=args.dry_run,
    )
    print("Dry-run plan (no PDFs or ChromaDB records changed)." if args.dry_run else "Ingestion run finished.")
    for key, value in result.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    main()