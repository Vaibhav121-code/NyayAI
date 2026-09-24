# NyaAI – AI-Assisted Legal Case Analysis and Precedent Retrieval

NyaAI is a college project focused on building the foundation for an AI-assisted legal research system for Indian Supreme Court judgments. The current version is not a finished legal research assistant. It is a Retrieval-Augmented Generation (RAG) pipeline foundation that can ingest local PDF judgments, clean and chunk them, generate embeddings, store them in ChromaDB, and retrieve semantically relevant passages.

The current version does not generate legal advice or verdict predictions. It only retrieves relevant passages from the supplied judgment corpus.

## 1. What NyaAI is

NyaAI is meant to eventually support the following workflow:

1. A user provides a description of a legal issue or case.
2. The system retrieves relevant Indian Supreme Court judgments.
3. Judgments are re-ranked and contextualized.
4. An LLM is given retrieved passages for a legal research brief.
5. The system identifies applicable legal provisions and compares facts.

At this stage, the project focuses only on the retrieval foundation:

PDF
→ text extraction
→ cleaning
→ chunking
→ embeddings
→ ChromaDB
→ semantic retrieval

## 2. Current scope

This project currently includes:

- reading local PDF judgments with PyMuPDF
- extracting text from each page
- conservative cleaning of judgment text
- word-aware chunking with overlap
- embedding generation using sentence-transformers
- persistent ChromaDB storage
- semantic search over the embedded chunks
- dataset inspection utilities

The following are intentionally not part of this version:

- React frontend
- Node.js backend
- MongoDB
- LLM generation
- LangChain
- cross-encoder reranking
- statutory mapping
- evidence-gap analysis
- authentication
- deployment
- automated legal conclusions

## 3. RAG architecture

The current architecture is intentionally simple and beginner-readable:

- `rag/pdf_loader.py`: loads PDFs and extracts text while preserving page structure
- `rag/cleaner.py`: removes obvious formatting noise without aggressively altering legal content
- `rag/chunker.py`: splits the cleaned judgment into chunks of roughly 700 words with 100-word overlap
- `rag/embeddings.py`: loads the embedding model and creates vector representations
- `rag/vector_store.py`: stores chunks in ChromaDB with source metadata
- `rag/search.py`: embeds a query and retrieves the top matching judgment chunks
- `rag/ingest.py`: orchestrates the entire ingestion pipeline

This intentionally avoids abstraction layers such as LangChain so that each component remains easy to study and explain in interviews or viva.

## 4. Dataset source

The target dataset is the Kaggle dataset:

https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024

Dataset name: "Legal Dataset: SC Judgments India 1950-2024"

The system is designed to work with local PDFs that the user manually downloads and places in `data/raw/`. This project does not automatically download the dataset.

## 5. Project structure

```text
NyaAI/
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── pdf_loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── ingest.py
│   └── search.py
├── scripts/
│   └── inspect_dataset.py
├── requirements.txt
├── .gitignore
├── README.md
├── .env.example
└── tests/
```

## 6. Installation

Create a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 7. How to add PDFs

Place judgment PDFs manually in the `data/raw/` directory.

The project is intended to work with a smaller development set first, such as 10–20 PDFs, before scaling to the full Kaggle corpus.

## 8. How to inspect the dataset

Run:

```bash
python scripts/inspect_dataset.py
```

This script will:

- find all PDFs recursively in `data/raw/`
- list each file and file size
- attempt to open each PDF with PyMuPDF
- report page count and extractability
- show a summary of readable vs unreadable PDFs

## 9. How to ingest documents

Before ingestion, ensure your PDFs are inside `data/raw/`.

Then run:

```bash
python -m rag.ingest
```

This will:

1. discover all PDFs
2. extract text
3. clean text
4. chunk text
5. generate embeddings
6. insert chunks into ChromaDB

## 10. How semantic search works

Run:

```bash
python -m rag.search
```

Then type a natural-language legal query, for example:

```text
A person allegedly obtained money through a false promise.
```

The program will:

- embed the query
- send it to ChromaDB
- retrieve the top matching chunks
- print each result with its document name, chunk ID, similarity score, relevant text, and source metadata

## 11. Example query

```text
A person allegedly obtained money through a false promise.
```

This returns the most semantically similar judgment passages from the local PDF corpus.

## 12. Current limitations

This version is intentionally limited.

- no LLM-based answer generation
- no legal reasoning engine
- no statutory mapping
- no cross-encoder reranking
- no authentication or UI
- no automatic legal advice or verdict prediction
- no OCR support yet
- limited metadata extraction beyond file name and chunk IDs

## 13. Future roadmap

Planned future phases include:

- OCR for scanned judgments
- legal metadata extraction (case names, dates, parties, citations)
- improved chunking and re-ranking
- LLM-generated legal research briefs
- statutory provision mapping
- evidence-gap analysis
- user auth and dashboard
- frontend integration
- deployment and scaling

## 14. Current pipeline

The current version does the following:

PDF
→ text extraction
→ cleaning
→ chunking
→ embeddings
→ ChromaDB
→ semantic retrieval

This is a stripped-down retrieval foundation for legal judgment search and is intentionally not a production-ready legal assistant.

## 15. Notes

- The application expects local PDF files supplied by the user.
- No secrets are committed.
- The project uses environment variables via `.env` and `.env.example`.
- If no PDFs are placed in `data/raw/`, the system will report that no PDFs are available for ingestion or search.
