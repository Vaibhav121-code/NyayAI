from __future__ import annotations

import argparse
import re
import json
import time
from pathlib import Path

from rag.config import INGESTION_COLLECTION_NAME, TOP_K
from rag.search import run_search

QUERIES = [
    (
        "Constitutional law",
        "How does the basic structure doctrine limit Parliament's power to amend the Constitution under Article 368?",
    ),
    (
        "Constitutional law",
        "How do Articles 13 and 368 interact when a constitutional amendment affects fundamental rights?",
    ),
    (
        "Criminal law",
        "What is the presumption of innocence and what standard of proof applies in a criminal trial?",
    ),
    (
        "Criminal law",
        "When may a criminal conviction be based on circumstantial evidence?",
    ),
    (
        "Contract law",
        "When can coercion, misrepresentation, or undue influence make an agreement voidable under the Indian Contract Act?",
    ),
    (
        "Contract law",
        "When does supervening impossibility frustrate a contract under section 56 of the Indian Contract Act?",
    ),
    (
        "Evidence law",
        "How should a court assess the reliability and corroboration of a dying declaration?",
    ),
    (
        "Evidence law",
        "How is the burden of proof allocated under the Indian Evidence Act?",
    ),
    (
        "Fundamental rights",
        "What restrictions on freedom of speech are permitted by Article 19(2)?",
    ),
    (
        "Fundamental rights",
        "How does the Supreme Court assess deprivation of personal liberty under Article 21?",
    ),
]

STOP_WORDS = {
    "a", "an", "and", "are", "does", "how", "in", "is", "of", "on", "the", "to", "under", "what", "when",
}


def _query_terms(query: str) -> set[str]:
    return {
        term.casefold()
        for term in re.findall(r"[A-Za-z0-9]+", query)
        if len(term) > 2 and term.casefold() not in STOP_WORDS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only retrieval evaluation for Indian legal queries.")
    parser.add_argument("--collection", default=INGESTION_COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--excerpt-chars", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("evaluation/retrieval_evaluation.json"))
    args = parser.parse_args()
    if args.top_k <= 0 or args.excerpt_chars <= 0:
        parser.error("--top-k and --excerpt-chars must be positive")

    report = {
        "collection": args.collection,
        "top_k": args.top_k,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "measurement": "diagnostic only; no ground-truth relevance labels",
        "queries": [],
    }
    latencies = []
    print(f"Collection: {args.collection}")
    print("Query-term overlap is a lexical diagnostic, not a relevance label or legal correctness assessment.")
    for category, query in QUERIES:
        started = time.perf_counter()
        response = run_search(query, top_k=args.top_k, collection_name=args.collection)
        latency = time.perf_counter() - started
        latencies.append(latency)
        query_terms = _query_terms(query)
        results = response["results"]
        unique_documents = len({
            str(result["metadata"].get("document_id") or result["document"])
            for result in results
        })
        metadata_complete = sum(
            1 for result in results
            if result["metadata"].get("file_name")
            and result["metadata"].get("page_start") is not None
            and result["metadata"].get("chunk_id")
        )
        report["queries"].append({
            "category": category,
            "query": query,
            "latency_seconds": round(latency, 4),
            "candidates_retrieved": response["candidates_retrieved"],
            "duplicates_removed": response["duplicates_removed"],
            "unique_results": len(results),
            "unique_documents": unique_documents,
            "metadata_complete_results": metadata_complete,
        })
        print(f"\n=== {category}: {query}")
        print(
            f"Candidates: {response['candidates_retrieved']}; "
            f"duplicate passages removed: {response['duplicates_removed']}; "
            f"unique results: {len(results)}"
        )
        if not results:
            print("No results returned.")
            continue

        for result in results:
            metadata = result["metadata"]
            text = str(result["text"])
            text_terms = {term.casefold() for term in re.findall(r"[A-Za-z0-9]+", text)}
            shared_terms = sorted(query_terms & text_terms)
            print(
                f"\n[{result['rank']}] {result['document']} | "
                f"pages {metadata.get('page_start')}-{metadata.get('page_end')} | "
                f"distance={result.get('distance')} | similarity={result.get('similarity')}"
            )
            print(
                "metadata="
                f"document_id={metadata.get('document_id')}; "
                f"chunk_id={metadata.get('chunk_id')}; "
                f"positions={metadata.get('start_position')}-{metadata.get('end_position')}"
            )
            print(f"query terms present={shared_terms or 'none'}")
            print("passage=" + " ".join(text.split())[:args.excerpt_chars])

    print("\nEvaluation note: inspect excerpts for topic fit; no ground-truth labels are applied.")
    report["average_latency_seconds"] = round(sum(latencies) / len(latencies), 4) if latencies else None
    report["total_latency_seconds"] = round(sum(latencies), 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nStructured report: {args.output}")


if __name__ == "__main__":
    main()