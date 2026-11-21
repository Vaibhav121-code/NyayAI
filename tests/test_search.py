from types import SimpleNamespace

from rag import search
from rag.config import INGESTION_COLLECTION_NAME


def test_search_defaults_to_v2_and_allows_legacy_selection(monkeypatch):
    requested_collections = []
    collection = SimpleNamespace(count=lambda: 100)
    monkeypatch.setattr(
        search,
        "initialize_collection",
        lambda collection_name: requested_collections.append(collection_name) or collection,
    )
    monkeypatch.setattr(search, "embed_query", lambda _: [0.1, 0.2])
    monkeypatch.setattr(
        search,
        "vector_search",
        lambda *_args, **_kwargs: {"documents": [[]], "metadatas": [[]], "distances": [[]]},
    )

    search.run_search("constitutional law")
    search.run_search("constitutional law", collection_name="nyaai_judgments")

    assert requested_collections == [INGESTION_COLLECTION_NAME, "nyaai_judgments"]


def test_search_skips_embedding_when_collection_is_empty(monkeypatch):
    collection = SimpleNamespace(count=lambda: 0)
    monkeypatch.setattr(search, "initialize_collection", lambda **_: collection)
    monkeypatch.setattr(
        search,
        "embed_query",
        lambda _: (_ for _ in ()).throw(AssertionError("embedding should be skipped")),
    )

    response = search.run_search("legal query")

    assert response["results"] == []
    assert response["corpus_available"] is False


def test_search_overfetches_before_deduplication(monkeypatch):
    collection = SimpleNamespace(count=lambda: 100)
    monkeypatch.setattr(search, "initialize_collection", lambda **_: collection)
    monkeypatch.setattr(search, "embed_query", lambda _: [0.1, 0.2])
    requested_counts = []
    passage = "A distinct legal passage containing enough words to be returned in search results."

    def fake_search(_collection, _embedding, top_k):
        requested_counts.append(top_k)
        texts = [passage, passage, "A second passage from the same judgment with different information."]
        metadatas = [
            {"chunk_id": "first", "document_id": "doc-1"},
            {"chunk_id": "duplicate", "document_id": "doc-2"},
            {"chunk_id": "second", "document_id": "doc-1"},
        ]
        return {"documents": [texts], "metadatas": [metadatas], "distances": [[0.1, 0.2, 0.3]]}

    monkeypatch.setattr(search, "vector_search", fake_search)

    response = search.run_search("legal query", top_k=2)

    assert requested_counts == [10]
    assert [item["chunk_id"] for item in response["results"]] == ["first", "second"]
    assert response["duplicates_removed"] == 1
    assert response["results"][0]["distance"] == 0.1


def test_search_can_restrict_results_to_selected_documents(monkeypatch):
    requested_filters = []
    collection = SimpleNamespace(count=lambda: 25)
    monkeypatch.setattr(search, "initialize_collection", lambda **_: collection)
    monkeypatch.setattr(search, "embed_query", lambda _: [0.1, 0.2])

    def fake_search(_collection, _embedding, top_k, where=None):
        requested_filters.append((top_k, where))
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    monkeypatch.setattr(search, "vector_search", fake_search)

    search.run_search("follow-up question", top_k=2, document_ids=["doc-1", "doc-2"])

    expected_filter = {"document_id": {"$in": ["doc-1", "doc-2"]}}
    assert requested_filters == [(10, expected_filter)]


def test_vector_store_search_passes_document_filter():
    from rag.vector_store import search as vector_store_search

    requested = {}

    class Collection:
        def query(self, **kwargs):
            requested.update(kwargs)
            return {}

    document_filter = {"document_id": {"$in": ["doc-1"]}}
    vector_store_search(
        Collection(),
        SimpleNamespace(tolist=lambda: [0.1, 0.2]),
        top_k=3,
        where=document_filter,
    )

    assert requested["where"] == document_filter
    assert requested["n_results"] == 3


def test_deduplication_removes_near_duplicates_but_keeps_distinct_same_case_passages():
    base = " ".join(f"legalword{index}" for index in range(40))
    near_duplicate = base.replace("legalword20", "legalword20," )
    distinct = " ".join(f"differentterm{index}" for index in range(40))
    results = [
        {"chunk_id": "one", "text": base, "metadata": {"document_id": "same-case"}},
        {"chunk_id": "near", "text": near_duplicate, "metadata": {"document_id": "other-case"}},
        {"chunk_id": "distinct", "text": distinct, "metadata": {"document_id": "same-case"}},
    ]

    deduplicated = search.deduplicate_results(results)

    assert [item["chunk_id"] for item in deduplicated] == ["one", "distinct"]


def test_deduplication_handles_long_near_duplicate_passages():
    base = " ".join(["court"] * 300 + [f"principle{index}" for index in range(400)])
    near_duplicate = base.replace("principle200", "principle200,")
    distinct = " ".join(f"other{index}" for index in range(700))

    deduplicated = search.deduplicate_results([
        {"chunk_id": "base", "text": base},
        {"chunk_id": "near", "text": near_duplicate},
        {"chunk_id": "different", "text": distinct},
    ])

    assert [item["chunk_id"] for item in deduplicated] == ["base", "different"]


def test_run_search_rejects_empty_or_oversized_queries(monkeypatch):
    monkeypatch.setattr(search, "initialize_collection", lambda **_: SimpleNamespace())

    try:
        search.run_search("   ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Empty query was accepted")

    try:
        search.run_search("x" * 4001)
    except ValueError as exc:
        assert "4000" in str(exc)
    else:
        raise AssertionError("Oversized query was accepted")