import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def authenticated_client(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))
    with TestClient(main.app) as client:
        response = client.post(
            "/api/auth/signup",
            json={"email": "researcher@example.com", "password": "research-password-123"},
        )
        assert response.status_code == 201
        yield client


def test_health_endpoint_reports_collection(monkeypatch):
    class Collection:
        def count(self):
            return 25

    monkeypatch.setattr(main, "initialize_collection", lambda _name: Collection())
    response = TestClient(main.app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["chunk_count"] == 25


def test_status_declares_retrieval_only_mode(monkeypatch, authenticated_client):
    class Collection:
        def count(self):
            return 0

    monkeypatch.setattr(main, "initialize_collection", lambda _name: Collection())
    response = authenticated_client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["mode"] == "retrieval_only"
    assert response.json()["corpus_available"] is False
    assert response.json()["chunk_count"] == 0
    assert "llm_configured" not in response.json()


def test_query_returns_only_retrieved_sources(monkeypatch, authenticated_client):
    requested = {}

    def fake_search(query, top_k, collection_name):
        requested.update(query=query, top_k=top_k, collection_name=collection_name)
        return {
            "results": [{
                "document": "Example_Case_on_1_January_2020.pdf",
                "chunk_id": "chunk-123",
                "text": "Retrieved passage from the judgment.",
                "distance": 0.2,
                "similarity": 0.83,
                "metadata": {
                    "file_name": "Example_Case_on_1_January_2020.pdf",
                    "page_start": 12,
                    "page_end": 12,
                    "chunk_id": "chunk-123",
                    "document_id": "doc-1",
                },
            }],
            "candidates_retrieved": 10,
            "duplicates_removed": 1,
        }

    monkeypatch.setattr(main, "run_search", fake_search)
    response = authenticated_client.post(
        "/api/query",
        json={"query": "Article 21 personal liberty", "top_k": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert requested["query"] == "Article 21 personal liberty"
    assert requested["top_k"] == 1
    assert body["mode"] == "retrieval_only"
    assert body["results"][0]["chunk_id"] == "chunk-123"
    assert body["results"][0]["page_start"] == 12
    assert body["results"][0]["excerpt"] == "Retrieved passage from the judgment."
    assert body["results"][0]["case_name"] == "Example Case"
    assert body["results"][0]["year"] == "2020"
    assert body["results"][0]["decision_date"] == "1 January 2020"
    assert body["results"][0]["document_id"] == "doc-1"
    assert body["duplicates_removed"] == 1
    assert "direct_answer" not in body
    assert "legal_principles" not in body


def test_query_can_restrict_follow_up_search_to_selected_cases(monkeypatch, authenticated_client):
    requested = {}

    def fake_search(query, top_k, collection_name, document_ids=None):
        requested.update(
            query=query,
            top_k=top_k,
            collection_name=collection_name,
            document_ids=document_ids,
        )
        return {"results": [], "candidates_retrieved": 0, "duplicates_removed": 0}

    monkeypatch.setattr(main, "run_search", fake_search)
    response = authenticated_client.post(
        "/api/query",
        json={
            "query": "follow-up question",
            "document_ids": ["doc-1", "doc-2"],
        },
    )

    assert response.status_code == 200
    assert requested["document_ids"] == ["doc-1", "doc-2"]


def test_query_reports_no_results_without_claiming_no_authorities(monkeypatch, authenticated_client):
    monkeypatch.setattr(main, "run_search", lambda *_args, **_kwargs: {
        "results": [],
        "candidates_retrieved": 0,
        "duplicates_removed": 0,
    })
    response = authenticated_client.post(
        "/api/query",
        json={"query": "a topic not represented in this corpus"},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert "does not establish" in response.json()["limitations"][0]


def test_query_does_not_invent_date_when_filename_has_no_date(monkeypatch, authenticated_client):
    monkeypatch.setattr(main, "run_search", lambda *_args, **_kwargs: {
        "results": [{
            "document": "A_Case_vs_State_1.pdf",
            "chunk_id": "chunk-no-date",
            "text": "A passage.",
            "distance": 0.3,
            "similarity": 0.77,
            "metadata": {"document_id": "doc-no-date"},
        }],
        "candidates_retrieved": 1,
        "duplicates_removed": 0,
    })

    response = authenticated_client.post("/api/query", json={"query": "case summary"})

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["case_name"] == "A Case vs State"
    assert result["year"] is None
    assert result["decision_date"] is None


def test_query_validates_blank_query_and_top_k(authenticated_client):
    blank = authenticated_client.post("/api/query", json={"query": "   "})
    excessive_k = authenticated_client.post("/api/query", json={"query": "constitutional law", "top_k": 21})
    assert blank.status_code == 422
    assert excessive_k.status_code == 422


def test_case_analysis_returns_retrieval_comparison_without_a_model(monkeypatch, authenticated_client):
    requested = {}

    def fake_search(query, top_k, collection_name, document_ids=None):
        requested.update(
            query=query,
            top_k=top_k,
            collection_name=collection_name,
            document_ids=document_ids,
        )
        return {
            "results": [{
                "document": "Example_Case_on_1_January_2020.pdf",
                "text": "The prosecution must establish a complete chain of circumstantial evidence.",
                "metadata": {
                    "document_id": "doc-1",
                    "page_start": 12,
                    "page_end": 12,
                    "case_name": "Example Case",
                },
                "similarity": 0.83,
            }],
            "candidates_retrieved": 1,
            "duplicates_removed": 0,
        }

    monkeypatch.setattr(main, "run_search", fake_search)

    response = authenticated_client.post(
        "/api/analyze-case",
        json={
            "case_summary": "The prosecution must prove a circumstantial chain beyond reasonable doubt.",
            "question": "Suggest arguments for this case.",
            "document_ids": ["doc-1"],
        },
    )

    assert response.status_code == 200
    assert requested["query"].startswith("Suggest arguments for this case.")
    assert requested["top_k"] == 5
    assert requested["document_ids"] == ["doc-1"]
    body = response.json()
    assert body["method"] == "retrieval_comparison"
    assert "model" not in body
    assert body["analysis"]["case_summary"].startswith("The prosecution must prove")
    assert body["analysis"]["comparisons"] == [{
        "source_id": "S1",
        "shared_terms": ["chain", "circumstantial", "prosecution"],
        "summary_only_terms": ["beyond", "doubt", "prove", "reasonable"],
        "passage_only_terms": ["complete", "establish", "evidence"],
    }]
    assert body["analysis"]["review_questions"]
    assert body["sources"][0]["source_id"] == "S1"
    assert body["sources"][0]["pages"] == "12"
    assert body["sources"][0]["retrieval_similarity"] == 0.83
    assert "No language model is used" in body["notice"]


def test_case_analysis_returns_not_found_without_retrieved_sources(monkeypatch, authenticated_client):
    monkeypatch.setattr(main, "run_search", lambda *_args, **_kwargs: {"results": []})

    response = authenticated_client.post(
        "/api/analyze-case",
        json={
            "case_summary": "A case summary.",
            "question": "Suggest arguments.",
            "document_ids": ["doc-1"],
        },
    )

    assert response.status_code == 404
    assert "No relevant passages" in response.json()["detail"]
