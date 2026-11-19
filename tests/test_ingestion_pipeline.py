import csv
import hashlib
import uuid
from pathlib import Path

import chromadb
import numpy as np
import pytest

from rag import ingest
from rag.chunker import chunk_pages
from rag.ingestion_tracker import IngestionTracker
from rag.vector_store import add_documents


@pytest.fixture
def temporary_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(name=f"test_{uuid.uuid4().hex}")
    yield collection
    client.close()


def _write_pdf(path: Path, content: bytes = b"pdf contents") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _install_pipeline_mocks(monkeypatch, collection, extract_text):
    monkeypatch.setattr(ingest, "initialize_collection", lambda **_: collection)
    monkeypatch.setattr(ingest, "extract_text", extract_text)
    monkeypatch.setattr(
        ingest,
        "embed_texts",
        lambda texts: np.ones((len(texts), 3), dtype=np.float32),
    )


def test_chunk_ids_are_deterministic():
    pages = [{"page_number": 4, "text": "legal judgment text"}]
    first = chunk_pages(pages, "document", "case.pdf")
    second = chunk_pages(pages, "document", "case.pdf")
    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]


def test_vector_store_batches_use_metadata_ids_and_upsert(temporary_collection):
    first_ids = ["chunk-a", "chunk-b"]
    add_documents(
        temporary_collection,
        ["a", "b"],
        np.ones((2, 3), dtype=np.float32),
        [{"chunk_id": chunk_id} for chunk_id in first_ids],
    )
    add_documents(
        temporary_collection,
        ["c", "d"],
        np.ones((2, 3), dtype=np.float32),
        [{"chunk_id": "chunk-c"}, {"chunk_id": "chunk-d"}],
    )
    add_documents(
        temporary_collection,
        ["updated a"],
        np.ones((1, 3), dtype=np.float32),
        [{"chunk_id": "chunk-a"}],
    )

    assert temporary_collection.count() == 4
    assert temporary_collection.get(ids=["chunk-a"])["documents"] == ["updated a"]

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        add_documents(
            temporary_collection,
            ["x", "y"],
            np.ones((2, 3), dtype=np.float32),
            [{"chunk_id": "same"}, {"chunk_id": "same"}],
        )

    with pytest.raises(ValueError, match="equal lengths"):
        add_documents(temporary_collection, ["x"], np.ones((0, 3)), [])


def test_content_hash_is_streamed_sha256(tmp_path):
    pdf_path = _write_pdf(tmp_path / "case.pdf", b"a" * 37)
    assert ingest.calculate_content_hash(pdf_path, block_size=5) == hashlib.sha256(b"a" * 37).hexdigest()


def test_embeddings_use_configured_batch_and_float32(monkeypatch):
    from rag import embeddings

    class Model:
        def encode(self, texts, **kwargs):
            self.kwargs = kwargs
            return np.ones((len(texts), 3), dtype=np.float64)

    model = Model()
    monkeypatch.setattr(embeddings, "EMBEDDING_BATCH_SIZE", 7)
    monkeypatch.setattr(embeddings, "load_embedding_model", lambda: model)

    result = embeddings.embed_texts(["one", "two"])

    assert result.dtype == np.float32
    assert model.kwargs["batch_size"] == 7
    assert model.kwargs["normalize_embeddings"] is True


def test_embedding_batch_size_must_be_positive(monkeypatch):
    from rag import embeddings

    monkeypatch.setattr(embeddings, "EMBEDDING_BATCH_SIZE", 0)
    with pytest.raises(ValueError, match="must be positive"):
        embeddings.embed_texts([])


def test_process_pdf_cleans_each_page_and_preserves_page_numbers(tmp_path, monkeypatch):
    pdf_path = _write_pdf(tmp_path / "2020" / "case.pdf")
    monkeypatch.setattr(
        ingest,
        "extract_text",
        lambda _: {
            "error": None,
            "extractable": True,
            "pages": [
                {"page_number": 3, "text": " First   page. "},
                {"page_number": 8, "text": "Second\n\npage."},
            ],
        },
    )

    texts, chunks, metadata = ingest.process_pdf(pdf_path, data_dir=tmp_path, content_hash="digest")

    assert texts == ["First page.", "Second\n\npage."]
    assert {item["page_start"] for item in metadata} == {3, 8}
    assert all(item["page_start"] == item["page_end"] for item in metadata)
    assert all(item["content_hash"] == "digest" for item in metadata)
    assert [chunk["chunk_id"] for chunk in chunks] == [item["chunk_id"] for item in metadata]
    assert ingest.stable_document_id(pdf_path, tmp_path) == ingest.stable_document_id(pdf_path, tmp_path)


def test_sqlite_tracker_persists_document_states(tmp_path):
    tracker = IngestionTracker(tmp_path / "tracker.sqlite3")
    assert tracker.get("case.pdf") is None

    tracker.mark_processing("case.pdf", "hash-1")
    assert tracker.get("case.pdf")["status"] == "processing"
    tracker.mark_completed("case.pdf", "hash-1", 7)
    assert tracker.get("case.pdf") == {
        "path": "case.pdf",
        "content_hash": "hash-1",
        "status": "completed",
        "chunk_count": 7,
        "error": None,
    }
    tracker.mark_failed("broken.pdf", "hash-2", "bad PDF")
    assert tracker.counts() == {"completed": 1, "failed": 1}


def test_resume_retries_interrupted_upsert_without_duplicate_ids(tmp_path, monkeypatch, temporary_collection):
    root = tmp_path / "raw"
    pdf_path = _write_pdf(root / "case.pdf")
    text = " ".join(["judgment"] * 800)
    _install_pipeline_mocks(
        monkeypatch,
        temporary_collection,
        lambda _: {"error": None, "extractable": True, "pages": [{"page_number": 6, "text": text}]},
    )

    original_tracker = ingest.IngestionTracker
    state = {"interrupt": True}

    class InterruptOnceTracker(original_tracker):
        def mark_completed(self, path, content_hash, chunk_count):
            if state["interrupt"]:
                state["interrupt"] = False
                raise RuntimeError("simulated interruption after indexing")
            super().mark_completed(path, content_hash, chunk_count)

    monkeypatch.setattr(ingest, "IngestionTracker", InterruptOnceTracker)
    kwargs = {
        "data_dir": root,
        "collection_name": "temporary",
        "tracking_db": tmp_path / "tracker.sqlite3",
        "failed_log": tmp_path / "failures.csv",
    }

    first = ingest.ingest_documents(**kwargs)
    assert first["documents_failed"] == 1
    count_after_interruption = temporary_collection.count()
    assert count_after_interruption > 0

    second = ingest.ingest_documents(resume=True, **kwargs)
    assert second["documents_processed"] == 1
    assert temporary_collection.count() == count_after_interruption
    assert IngestionTracker(kwargs["tracking_db"]).get("case.pdf")["status"] == "completed"


def test_changed_pdf_is_reindexed_before_old_chunks_are_removed(tmp_path, monkeypatch, temporary_collection):
    root = tmp_path / "raw"
    pdf_path = _write_pdf(root / "case.pdf", b"version one")
    _install_pipeline_mocks(
        monkeypatch,
        temporary_collection,
        lambda path: {
            "error": None,
            "extractable": True,
            "pages": [{"page_number": 2, "text": " ".join([path.read_bytes().decode()] * 800)}],
        },
    )
    kwargs = {
        "data_dir": root,
        "collection_name": "temporary",
        "tracking_db": tmp_path / "tracker.sqlite3",
        "failed_log": tmp_path / "failures.csv",
    }

    ingest.ingest_documents(**kwargs)
    old_records = temporary_collection.get(include=["metadatas"])
    old_ids = set(old_records["ids"])
    old_hash = old_records["metadatas"][0]["content_hash"]
    pdf_path.write_bytes(b"version two")

    result = ingest.ingest_documents(resume=True, **kwargs)
    new_records = temporary_collection.get(include=["metadatas"])

    assert result["documents_processed"] == 1
    assert old_hash != new_records["metadatas"][0]["content_hash"]
    assert old_ids.isdisjoint(set(new_records["ids"]))
    assert all(item["content_hash"] == ingest.calculate_content_hash(pdf_path) for item in new_records["metadatas"])


def test_limit_and_csv_failure_logging_do_not_stop_the_run(tmp_path, monkeypatch, temporary_collection):
    root = tmp_path / "raw"
    _write_pdf(root / "a-broken.pdf", b"broken PDF bytes")
    _write_pdf(root / "b-good.pdf", b"good PDF bytes")
    text = "A judgment with enough extractable text."

    def extract(path):
        if path.name.startswith("a-"):
            return {
                "error": 'invalid, "scanned"\nPDF',
                "extractable": False,
                "pages": [],
            }
        return {"error": None, "extractable": True, "pages": [{"page_number": 4, "text": text}]}

    _install_pipeline_mocks(monkeypatch, temporary_collection, extract)
    kwargs = {
        "data_dir": root,
        "collection_name": "temporary",
        "tracking_db": tmp_path / "tracker.sqlite3",
        "failed_log": tmp_path / "failures.csv",
    }

    limited = ingest.ingest_documents(limit=1, **kwargs)
    assert limited["documents_discovered"] == 2
    assert limited["documents_selected"] == 1
    assert limited["documents_failed"] == 1
    assert temporary_collection.count() == 0

    complete = ingest.ingest_documents(**kwargs)
    assert complete["documents_processed"] == 1
    assert complete["documents_failed"] == 1
    with Path(kwargs["failed_log"]).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    assert len(rows) == 3
    assert rows[1][0] == "a-broken.pdf"
    assert '"scanned"' in rows[1][2]
    assert "\n" in rows[1][2]


def test_unextractable_pdf_is_recorded_as_failed_without_ocr(tmp_path, monkeypatch, temporary_collection):
    root = tmp_path / "raw"
    _write_pdf(root / "empty.pdf")
    _install_pipeline_mocks(
        monkeypatch,
        temporary_collection,
        lambda _: {"error": None, "extractable": False, "pages": []},
    )

    result = ingest.ingest_documents(
        data_dir=root,
        collection_name="temporary",
        tracking_db=tmp_path / "tracker.sqlite3",
        failed_log=tmp_path / "failures.csv",
    )

    assert result["documents_failed"] == 1
    assert "OCR was not attempted" in IngestionTracker(tmp_path / "tracker.sqlite3").get("empty.pdf")["error"]


def test_canonical_source_selection_keeps_one_path_without_changing_pdfs(tmp_path):
    root = tmp_path / "raw"
    first = _write_pdf(root / "case.pdf", b"identical source bytes")
    duplicate = _write_pdf(root / "archive" / "case.pdf", b"identical source bytes")
    unique = _write_pdf(root / "other.pdf", b"different bytes and size")
    original_bytes = {path: path.read_bytes() for path in (first, duplicate, unique)}

    sources = ingest.select_canonical_sources(
        ingest.discover_pdfs(root),
        data_dir=root,
    )

    assert len(sources.duplicate_groups) == 1
    assert sources.duplicate_groups[0] == [duplicate, first]
    assert [group[0] for group in sources.groups] == [duplicate, unique]
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes


def test_resume_limit_selects_pending_canonical_pdfs_and_dry_run_is_read_only(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    pdfs = [
        _write_pdf(root / f"{index:02}.pdf", bytes([index]) * (index + 1))
        for index in range(5)
    ]
    tracker_path = tmp_path / "tracker.sqlite3"
    tracker = IngestionTracker(tracker_path)

    class ReadOnlyCollection:
        def __init__(self):
            self.records = {}

        def get(self, where, include):
            document_id = where["document_id"]
            ids, metadatas = self.records.get(document_id, ([], []))
            return {"ids": ids, "metadatas": metadatas}

    collection = ReadOnlyCollection()

    def mark_indexed(path):
        relative = path.relative_to(root).as_posix()
        content_hash = ingest.calculate_content_hash(path)
        document_id = ingest.stable_document_id(path, root)
        tracker.mark_completed(relative, content_hash, 1)
        collection.records[document_id] = (
            [f"chunk-{document_id}"],
            [{"content_hash": content_hash}],
        )

    mark_indexed(pdfs[0])
    mark_indexed(pdfs[1])
    monkeypatch.setattr(ingest, "_open_existing_collection", lambda *_: collection)
    monkeypatch.setattr(
        ingest,
        "initialize_collection",
        lambda **_: pytest.fail("dry-run must not initialize or create a collection"),
    )
    tracker_bytes_before = tracker_path.read_bytes()
    pdf_bytes_before = {path: path.read_bytes() for path in pdfs}

    first_plan = ingest.ingest_documents(
        data_dir=root,
        limit=2,
        resume=True,
        dry_run=True,
        tracking_db=tracker_path,
        chroma_dir=tmp_path / "chroma",
    )

    assert first_plan["documents_discovered"] == 5
    assert first_plan["exact_duplicate_groups"] == 0
    assert first_plan["canonical_pdfs_selected"] == 5
    assert first_plan["already_completed_pdfs"] == 2
    assert first_plan["already_completed_canonical_sources"] == 2
    assert first_plan["remaining_pdfs"] == 3
    assert first_plan["would_process_next_batch"] == 2
    assert tracker_path.read_bytes() == tracker_bytes_before
    assert {path: path.read_bytes() for path in pdfs} == pdf_bytes_before

    mark_indexed(pdfs[2])
    mark_indexed(pdfs[3])
    second_plan = ingest.ingest_documents(
        data_dir=root,
        limit=2,
        resume=True,
        dry_run=True,
        tracking_db=tracker_path,
        chroma_dir=tmp_path / "chroma",
    )

    assert second_plan["already_completed_pdfs"] == 4
    assert second_plan["already_completed_canonical_sources"] == 4
    assert second_plan["remaining_pdfs"] == 1
    assert second_plan["would_process_next_batch"] == 1


def test_resume_recognizes_completed_noncanonical_duplicate_alias(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    noncanonical = _write_pdf(root / "case.pdf", b"identical source")
    canonical = _write_pdf(root / "archive" / "case.pdf", b"identical source")
    tracker_path = tmp_path / "tracker.sqlite3"
    tracker = IngestionTracker(tracker_path)
    content_hash = ingest.calculate_content_hash(noncanonical)
    document_id = ingest.stable_document_id(noncanonical, root)
    tracker.mark_completed("case.pdf", content_hash, 1)

    class ReadOnlyCollection:
        def get(self, where, include):
            assert where["document_id"] == document_id
            return {
                "ids": ["indexed-alias-chunk"],
                "metadatas": [{"content_hash": content_hash}],
            }

    monkeypatch.setattr(ingest, "_open_existing_collection", lambda *_: ReadOnlyCollection())
    result = ingest.ingest_documents(
        data_dir=root,
        limit=1,
        resume=True,
        dry_run=True,
        tracking_db=tracker_path,
        chroma_dir=tmp_path / "chroma",
    )

    assert canonical != noncanonical
    assert result["exact_duplicate_groups"] == 1
    assert result["canonical_pdfs_selected"] == 1
    assert result["already_completed_pdfs"] == 1
    assert result["already_completed_canonical_sources"] == 1
    assert result["remaining_pdfs"] == 0
    assert result["would_process_next_batch"] == 0