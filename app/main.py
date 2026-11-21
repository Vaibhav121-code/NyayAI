from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.retrieval_analysis import build_retrieval_analysis
from rag.config import CHROMA_DIR, EMBEDDING_MODEL, INGESTION_COLLECTION_NAME
from rag.search import run_search
from rag.vector_store import initialize_collection

logger = logging.getLogger("nyay_ai")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Nyay AI",
    version="1.0.0",
    description="Evidence-grounded legal research for Indian Supreme Court judgments.",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _case_details(
    metadata: dict[str, Any],
    file_name: str,
) -> dict[str, Any]:
    stem = Path(file_name).stem
    date_match = re.search(
        r"_on_(\d{1,2})_([A-Za-z]+)_((?:19|20)\d{2})",
        stem,
        flags=re.IGNORECASE,
    )
    name = str(metadata.get("case_name") or "").strip()
    if not name:
        name = stem[:date_match.start()] if date_match else re.sub(r"_\d+$", "", stem)
        name = re.sub(r"_+", " ", name).strip()
        name = re.sub(r"\s+vs\s+", " vs ", name, flags=re.IGNORECASE)

    raw_year = metadata.get("year")
    year = str(raw_year) if raw_year is not None else (
        date_match.group(3) if date_match else None
    )
    decision_date = str(metadata.get("decision_date") or "").strip() or None
    if decision_date is None and date_match:
        decision_date = (
            f"{int(date_match.group(1))} "
            f"{date_match.group(2).capitalize()} {date_match.group(3)}"
        )

    return {
        "case_name": name or None,
        "year": year,
        "decision_date": decision_date,
    }


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: str | None = Field(default=None, max_length=200)
    document_ids: list[str] | None = Field(default=None, max_length=20)


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: str | None = Field(default=None, max_length=200)


class CaseAnalysisRequest(BaseModel):
    case_summary: str = Field(..., min_length=1, max_length=4000)
    question: str = Field(..., min_length=1, max_length=1000)
    document_ids: list[str] = Field(..., min_length=1, max_length=20)


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError):
    raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        collection_name = INGESTION_COLLECTION_NAME
        collection = initialize_collection(collection_name)
        return {
            "status": "ok",
            "collection": collection_name,
            "chunk_count": collection.count(),
        }
    except Exception as exc:
        logger.exception("Health check failed")
        return {"status": "degraded", "collection": INGESTION_COLLECTION_NAME, "error": str(exc)}


@app.get("/api/status")
def status() -> dict[str, Any]:
    chunk_count = initialize_collection(INGESTION_COLLECTION_NAME).count()
    return {
        "default_collection": INGESTION_COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chroma_dir": str(CHROMA_DIR),
        "mode": "retrieval_only",
        "chunk_count": chunk_count,
        "corpus_available": chunk_count > 0,
    }


@app.post("/api/retrieve")
def retrieve(request: RetrievalRequest) -> dict[str, Any]:
    collection_name = request.collection or INGESTION_COLLECTION_NAME
    try:
        response = run_search(
            request.query,
            top_k=request.top_k,
            collection_name=collection_name,
        )
        return {"query": request.query.strip(), "collection": collection_name, **response}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail="Retrieval is temporarily unavailable") from exc


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, Any]:
    collection_name = request.collection or INGESTION_COLLECTION_NAME
    try:
        response = run_search(
            request.query,
            top_k=request.top_k,
            collection_name=collection_name,
            **({"document_ids": request.document_ids} if request.document_ids is not None else {}),
        )
        return {
            "query": request.query.strip(),
            "collection": collection_name,
            "mode": "retrieval_only",
            "corpus_available": response.get(
                "corpus_available",
                bool(response["results"]),
            ),
            "notice": (
                "These are semantically retrieved passages, not a generated answer "
                "or a legal conclusion."
            ),
            "results": [
                {
                    "source_number": index,
                    **_case_details(
                        result["metadata"],
                        str(result["document"]),
                    ),
                    "file_name": result["document"],
                    "document_id": result["metadata"].get("document_id"),
                    "page_start": result["metadata"].get("page_start"),
                    "page_end": result["metadata"].get("page_end"),
                    "chunk_id": result["chunk_id"],
                    "retrieval_distance": result.get("distance"),
                    "retrieval_similarity": result.get("similarity"),
                    "excerpt": result["text"],
                }
                for index, result in enumerate(response["results"], start=1)
            ],
            "candidates_retrieved": response["candidates_retrieved"],
            "duplicates_removed": response["duplicates_removed"],
            "disclaimer": (
                "Nyay AI provides search results, not legal advice or legal analysis. "
                "Similarity scores are technical retrieval signals, not legal confidence "
                "or proof of relevance. Review the full judgment and applicable law."
            ),
            "limitations": (
                ["No matching passages were retrieved. This does not establish that no "
                 "relevant judgment or law exists."]
                if not response["results"]
                else [
                    "Results are limited to the indexed judgment corpus.",
                    "The system does not determine legal applicability, identify all "
                    "relevant laws, or assess missing evidence.",
                ]
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Judgment search failed")
        raise HTTPException(status_code=503, detail="Judgment search is temporarily unavailable") from exc


@app.post("/api/analyze-case")
def analyze_case(request: CaseAnalysisRequest) -> dict[str, Any]:
    retrieval_query = (
        f"{request.question.strip()}\n\n"
        f"Case facts supplied by the user:\n{request.case_summary.strip()}"
    )[:4000]
    try:
        search_response = run_search(
            retrieval_query,
            top_k=5,
            collection_name=INGESTION_COLLECTION_NAME,
            document_ids=request.document_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Case-analysis retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Judgment retrieval for case analysis is temporarily unavailable",
        ) from exc

    results = search_response["results"]
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant passages were found in the selected judgments; no case comparison is available.",
        )

    sources: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        file_name = str(result["document"])
        details = _case_details(metadata, file_name)
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        if page_start is None:
            pages = "not recorded"
        elif page_end is None or page_end == page_start:
            pages = str(page_start)
        else:
            pages = f"{page_start}-{page_end}"
        sources.append({
            "source_id": f"S{index}",
            "case_name": details["case_name"] or file_name,
            "decision_date": details["decision_date"] or "not available",
            "pages": pages,
            "excerpt": str(result["text"])[:1200],
            "retrieval_similarity": result.get("similarity"),
        })

    return {
        "analysis": build_retrieval_analysis(request.case_summary, sources),
        "method": "retrieval_comparison",
        "sources": sources,
        "notice": (
            "No language model is used. Shared and differing terms are exact keyword matches "
            "between your summary and retrieved excerpts; they do not establish legal similarity, "
            "a legal argument, or missing evidence. Review full judgments with qualified counsel."
        ),
    }


@app.get("/api/documents/{chunk_id}")
def document(chunk_id: str) -> dict[str, Any]:
    try:
        collection = initialize_collection(INGESTION_COLLECTION_NAME)
        record = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    except Exception as exc:
        logger.exception("Document lookup failed")
        raise HTTPException(status_code=503, detail="Source metadata is unavailable") from exc
    if not record.get("ids"):
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {
        "chunk_id": chunk_id,
        "text": record["documents"][0],
        "metadata": record["metadatas"][0],
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
