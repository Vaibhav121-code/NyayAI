import argparse
from difflib import SequenceMatcher
import logging

from rag.config import INGESTION_COLLECTION_NAME, TOP_K
from rag.embeddings import embed_query
from rag.vector_store import initialize_collection, search as vector_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _normalized_passage(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_near_duplicate(left: str, right: str) -> bool:
    if left == right:
        return True
    left_tokens = left.split()
    right_tokens = right.split()
    if min(len(left_tokens), len(right_tokens)) < 25:
        return False
    matcher = SequenceMatcher(None, left_tokens, right_tokens)
    return matcher.quick_ratio() >= 0.96 and matcher.ratio() >= 0.96


def deduplicate_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep the highest-ranked copy of each identical or near-identical passage."""
    unique_results: list[dict[str, object]] = []
    normalized_passages: list[str] = []

    for result in results:
        text = str(result.get("text", ""))
        normalized = _normalized_passage(text)
        if any(_is_near_duplicate(normalized, previous) for previous in normalized_passages):
            continue
        unique_results.append(result)
        normalized_passages.append(normalized)

    return unique_results


def run_search(
    query: str,
    top_k: int = TOP_K,
    collection_name: str = INGESTION_COLLECTION_NAME,
    document_ids: list[str] | None = None,
) -> dict[str, object]:
    """Run semantic retrieval for a legal query and return formatted results."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    query = query.strip()
    if len(query) > 4000:
        raise ValueError("query must be 4000 characters or fewer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if document_ids is not None and (
        not document_ids
        or any(
            not isinstance(document_id, str) or not document_id.strip()
            for document_id in document_ids
        )
    ):
        raise ValueError("document_ids must contain non-empty document IDs")

    collection = initialize_collection(collection_name=collection_name)
    collection_count = collection.count()
    if collection_count == 0:
        return {
            "results": [],
            "candidates_retrieved": 0,
            "duplicates_removed": 0,
            "corpus_available": False,
        }

    query_embedding = embed_query(query)
    where = (
        {"document_id": {"$in": list(dict.fromkeys(document_ids))}}
        if document_ids is not None
        else None
    )
    candidate_count = min(collection_count, max(top_k * 5, top_k))
    if candidate_count == 0:
        return {
            "results": [],
            "candidates_retrieved": 0,
            "duplicates_removed": 0,
            "corpus_available": True,
        }

    if where is None:
        results = vector_search(collection, query_embedding, top_k=candidate_count)
    else:
        results = vector_search(collection, query_embedding, top_k=candidate_count, where=where)

    output: list[dict[str, object]] = []
    if not results or not results.get("documents"):
        return {
            "results": output,
            "candidates_retrieved": 0,
            "duplicates_removed": 0,
            "corpus_available": True,
        }

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for idx, document in enumerate(docs):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        distance = distances[idx] if idx < len(distances) else None
        output.append({
            "rank": idx + 1,
            "document": metadata.get("file_name", "unknown"),
            "chunk_id": metadata.get("chunk_id", "unknown"),
            "distance": distance,
            "similarity": 1.0 / (1.0 + distance) if distance is not None else None,
            "text": document,
            "metadata": metadata,
        })

    unique_results = deduplicate_results(output)
    duplicates_removed = len(output) - len(unique_results)
    unique_results = unique_results[:top_k]
    for rank, result in enumerate(unique_results, start=1):
        result["rank"] = rank

    return {
        "results": unique_results,
        "candidates_retrieved": len(output),
        "duplicates_removed": duplicates_removed,
        "corpus_available": True,
    }


def interactive_search(collection_name: str = INGESTION_COLLECTION_NAME) -> None:
    """Interactive loop for querying the judgment corpus."""
    print("Enter legal query:")
    while True:
        try:
            query = input("> ")
        except EOFError:
            print("\nExiting.")
            break

        if query.strip().lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not query.strip():
            continue

        response = run_search(query, collection_name=collection_name)
        results = response["results"]
        if not results:
            print("No results found.")
            continue

        for index, result in enumerate(results, start=1):
            print(f"\n-------------------------------------")
            print(f"Result {index}")
            print("-------------------------------------")
            print(f"Document: {result['document']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Distance: {result['distance']:.4f}" if result.get("distance") is not None else "Distance: N/A")
            print(f"Similarity: {result['similarity']:.4f}" if result.get("similarity") is not None else "Similarity: N/A")
            print("\nRelevant text:")
            print(result["text"][:1200])
            print("\nMetadata:")
            print(f"Page: {result['metadata'].get('page_start')} - {result['metadata'].get('page_end')}")
            print(f"Source file: {result['metadata'].get('file_name')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed Indian Supreme Court judgments.")
    parser.add_argument(
        "--collection",
        default=INGESTION_COLLECTION_NAME,
        help=f"Chroma collection to search (default: {INGESTION_COLLECTION_NAME})",
    )
    args = parser.parse_args()
    interactive_search(collection_name=args.collection)


if __name__ == "__main__":
    main()
