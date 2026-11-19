# Nyay AI

Nyay AI is a local semantic search application for Indian Supreme Court judgments. It retrieves passages from an existing ChromaDB corpus and displays the source document, page, and technical retrieval score. It does not integrate or call an LLM and does not generate legal conclusions.

**Nyay AI is not legal advice or legal analysis.** Verify every authority and consult a qualified lawyer.

## Features

- Semantic retrieval over `nyaai_judgments_v2`
- Duplicate and near-duplicate passage removal
- Source metadata for document, chunk, and page references
- Configurable top-k retrieval
- Retrieval-only API
- No external LLM provider or API credentials
- Professional responsive web interface
- Grouped, bullet-point case results with available decision date, pages, sources, technical score, and selected verbatim passage highlights
- Copy-case-bullets and copy-source-reference actions
- On-demand, source-cited argument research outline with prompts to assess factual application and counterarguments; not a ready-to-file submission
- Read-only retrieval evaluation

## Architecture

```mermaid
flowchart LR
    UI[Web interface] --> API[FastAPI]
    API --> Search[RAG search pipeline]
    Search --> Embed[MiniLM query embedding]
    Embed --> Chroma[ChromaDB v2 collection]
    Chroma --> Search
    Search --> Results[Passages and metadata]
    Results --> API
    API --> UI
```

## Repository layout

```text
app/                 FastAPI backend and static frontend
rag/                 PDF, chunking, embedding, ingestion, and retrieval modules
scripts/             Dataset and retrieval utilities
tests/               Automated tests
data/chroma/         Existing persistent ChromaDB data
data/raw/            Source PDFs (not committed)
evaluation/          Retrieval evaluation reports
```

## Existing corpus

The production collection is `nyaai_judgments_v2`:

- approximately 314,870 indexed chunks
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- embedding dimensions: 384
- chunks: approximately 700 words
- overlap: 100 words
- top-k: configurable, default 5

The legacy `nyaai_judgments` collection remains available and untouched. Existing PDF, ChromaDB, and ingestion files must not be recreated or deleted for normal application use.

## Installation

Requires Python 3.12 on Windows.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

No API key or external generation service is required. Search queries are processed by the local embedding model and the local ChromaDB collection. The embedding model may need to be downloaded the first time it is run.

Enter a short case summary or legal issue in the web interface. Retrieved passages are grouped by judgment and presented as concise case cards with filename-derived case names/dates where available, page references, technical similarity, and selected verbatim excerpts. An optional argument research outline assembles relevant retrieved passages with case/page references and prompts for factual application and counterarguments; it does not assess legal suitability or create a ready-to-file submission. This is not a generated case summary: the system does not infer the decision, legal holding, applicable law, or evidence gaps. Review the complete source passages before drawing conclusions.

## Run locally

```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

API documentation is available at `http://127.0.0.1:8000/docs`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CHROMA_DIR` | `./data/chroma` | Persistent ChromaDB directory |
| `COLLECTION_NAME` | `nyaai_judgments_v2` | Default collection |
| `INGESTION_COLLECTION_NAME` | `nyaai_judgments_v2` | Production collection |
| `EMBEDDING_MODEL` | MiniLM L6 v2 | Query and document embedding model |
| `TOP_K` | `5` | Default retrieval count |
| `CORS_ORIGINS` | local URLs | Allowed frontend origins |

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Collection and process health |
| `GET` | `/api/status` | Retrieval configuration |
| `POST` | `/api/retrieve` | Retrieval-only debugging |
| `POST` | `/api/query` | Semantic judgment search with source passages |
| `GET` | `/api/documents/{chunk_id}` | Retrieve one chunk and metadata |

Example retrieval:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/query" -ContentType "application/json" -Body '{"query":"How does Article 21 protect personal liberty?","top_k":5}'
```

## Search CLI

```powershell
python -m rag.search
python -m rag.search --collection nyaai_judgments
```

## Retrieval evaluation

The evaluation command is read-only and does not modify ChromaDB or re-ingest documents:

```powershell
python -m scripts.evaluate_retrieval --top-k 5 --output evaluation/retrieval_evaluation.json
```

The suite covers basic structure, Articles 13/368/19/21, circumstantial evidence, presumption of innocence, dying declarations, burden of proof, contractual coercion, frustration, and natural justice topics as available.

The report contains latency, candidate counts, duplicate removal counts, document diversity, metadata completeness, and lexical overlap diagnostics. Lexical overlap and retrieval similarity are not legal relevance labels or correctness measurements. Precision, Recall, and MRR require a manually labeled relevance dataset and are not fabricated here.

## Ingestion

Normal application use does not ingest data. Existing ingestion remains resumable and safe:

```powershell
python -m rag.ingest --dry-run --resume --limit 1000
python -m rag.ingest --limit 500 --resume
```

Do not run ingestion against the production collection merely to test application changes.

## Testing

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
```

Tests use mocks and temporary Chroma collections. They do not write to the production collection.

## Security and limitations

- The application does not use external LLM APIs or API credentials.
- `.env` and local databases are excluded from Git.
- The corpus does not represent every Indian judgment or statute.
- Metadata extraction is filename/page based until richer judgment metadata is available.
- The system does not generate legal analysis or legal conclusions.
- It does not automatically identify evidence gaps, applicable statutes, or all relevant authorities.
- Retrieval similarity is not legal relevance or confidence.
- OCR is not attempted for image-only PDFs.
