import logging

from rag.config import TOP_K
from rag.embeddings import embed_query
from rag.vector_store import initialize_collection, search as vector_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_search(query: str, top_k: int = TOP_K) -> dict[str, object]:
    """Run semantic retrieval for a legal query and return formatted results."""
    collection = initialize_collection()
    query_embedding = embed_query(query)
    results = vector_search(collection, query_embedding, top_k=top_k)

    output: list[dict[str, object]] = []
    if not results or not results.get("documents"):
        return {"results": output}

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
            "similarity": 1.0 / (1.0 + distance) if distance is not None else None,
            "text": document,
            "metadata": metadata,
        })

    return {"results": output}


def interactive_search() -> None:
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

        response = run_search(query)
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
            print(f"Similarity: {result['similarity']:.4f}" if result.get("similarity") is not None else "Similarity: N/A")
            print("\nRelevant text:")
            print(result["text"][:1200])
            print("\nMetadata:")
            print(f"Page: {result['metadata'].get('page_start')} - {result['metadata'].get('page_end')}")
            print(f"Source file: {result['metadata'].get('file_name')}")


def main() -> None:
    interactive_search()


if __name__ == "__main__":
    main()
