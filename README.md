# MineIntel AI

A retrieval-augmented generation (RAG) platform for analysing mining accident reports. Upload incident reports as PDFs (digital or scanned), then ask questions in plain English. Answers are generated only from the uploaded documents and every answer links back to the passages it came from. If the documents do not contain the answer, the system says so instead of guessing.

## Features

- PDF ingestion with text extraction and automatic OCR for scanned pages
- Text cleaning and page-aware, overlapping chunking so every citation points to an exact page
- Sentence embeddings stored in a FAISS index for semantic search
- Grounded question answering with numbered citations (document name and page)
- Refusal when retrieval is weak: the language model is not called at all if no passage is relevant enough
- Optional LangGraph workflow that grades retrieval and rewrites weak queries before answering
- React interface for uploading, chatting and inspecting source passages
- PostgreSQL storage for documents, chunks and query history
- Everything configurable through environment variables; runs on free tiers

## Architecture

```
React (Vite)  --->  FastAPI routers  --->  services  --->  repositories  --->  PostgreSQL
                                              |
                                              +--> FAISS index (chunk id -> vector)
                                              +--> embedding model (local or Hugging Face API)
                                              +--> chat model (any OpenAI-compatible API or Ollama)
```

Ingestion:

```
PDF -> text per page (PyMuPDF) -> OCR if the page has no text layer (Tesseract)
    -> clean -> split into overlapping chunks -> embed -> save rows + vectors
```

Question answering:

```
question -> embed -> FAISS top-k -> drop chunks below the similarity threshold
         -> prompt with numbered passages -> chat model -> answer with [n] citations
```

Agentic mode (LangGraph):

```
retrieve -> grade --(relevant)--------------------------> generate
              |
              +--(weak, retries left)--> rewrite query --> retrieve
              +--(weak, no retries left)-----------------> generate (reports "not found")
```

## Project structure

```
backend/
  app/
    api/            dependency wiring
    core/           settings, logging, domain exceptions
    db/             SQLAlchemy base, session, models
    repositories/   database queries
    routers/        HTTP endpoints
    schemas/        request and response models
    services/       ingestion, chunking, embeddings, FAISS, retrieval, QA, LangGraph
  tests/
frontend/
  src/              React app
render.yaml         Render blueprint
```

## Tech stack

| Area | Choice |
| --- | --- |
| API | Python, FastAPI |
| Frontend | React, Vite |
| Orchestration | LangChain (LCEL chains), LangGraph |
| Vector search | FAISS (`IndexFlatIP` inside `IndexIDMap`) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers, or the Hugging Face Inference API |
| Chat model | Any OpenAI-compatible endpoint (Groq, OpenRouter, Hugging Face router, Ollama, vLLM) or native Ollama |
| OCR | Tesseract through PyMuPDF page rendering |
| Database | PostgreSQL (SQLite works for quick local runs and tests) |
| Hosting | Render (backend), Vercel (frontend) |

## Getting started

### Prerequisites

- Python 3.11 or newer
- Node.js 18 or newer
- PostgreSQL (or leave `DATABASE_URL` unset to use SQLite)
- Tesseract OCR for scanned PDFs: `sudo apt install tesseract-ocr` on Ubuntu, `brew install tesseract` on macOS
- An API key for a chat model. Groq offers a free tier; a local Ollama install needs no key.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set at least `LLM_API_KEY` (and `DATABASE_URL` if you are using PostgreSQL), then:

```bash
uvicorn app.main:app --reload
```

The API runs at http://localhost:8000 and interactive docs are at http://localhost:8000/docs. Tables are created on startup.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open http://localhost:5173.

## Configuration

All settings are read from environment variables (see `backend/.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./mineintel.db` | Database connection string |
| `DATA_DIR` | `./data` | Uploaded PDFs and the FAISS index |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking in characters |
| `OCR_ENABLED` | `true` | Run OCR on pages with little or no text |
| `EMBEDDING_PROVIDER` | `local` | `local` or `hf_api` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model name |
| `TOP_K` | `5` | Passages retrieved per question |
| `MIN_RELEVANCE_SCORE` | `0.25` | Minimum cosine similarity for a passage to be used |
| `MAX_QUERY_REWRITES` | `2` | Rewrite attempts in agentic mode |
| `LLM_PROVIDER` | `openai_compatible` | `openai_compatible` or `ollama` |
| `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` | Groq defaults | Chat model settings |

To switch models, change the variables only. For example, to use a local Ollama model: `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1`. The right similarity threshold depends on the embedding model, so tune `MIN_RELEVANCE_SCORE` against your own documents.

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service status and number of indexed chunks |
| `POST` | `/api/documents` | Upload a PDF (multipart field `file`) |
| `GET` | `/api/documents` | List documents |
| `DELETE` | `/api/documents/{id}` | Remove a document and its vectors |
| `POST` | `/api/query` | Ask a question: `{"question": "...", "top_k": 5, "agentic": false}` |
| `GET` | `/api/query/history` | Recent questions and answers |

Example response from `/api/query`:

```json
{
  "answer": "Most roof falls were attributed to unscaled loose strata [1][2].",
  "grounded": true,
  "sources": [
    {"index": 1, "document_name": "east-stope.pdf", "page_number": 3, "score": 0.71, "text": "..."}
  ]
}
```

## Design decisions

- **Page-level chunking.** Pages are chunked separately so each chunk has an exact page number for citations.
- **Cosine similarity with FAISS.** Embeddings are L2-normalised and stored in a flat inner-product index, which gives exact cosine search. At the scale of a document collection this is fast enough and avoids approximate-index tuning.
- **Embeddings stored in PostgreSQL.** Free hosting tiers have an ephemeral disk, so the FAISS file can disappear on restart. Each chunk's vector is also saved in the database and the index is rebuilt from it on startup without re-running the embedding model.
- **Refusal before generation.** If no retrieved passage reaches the similarity threshold, the answer is "not found" and the language model is never called. The prompt also tells the model to reply with a fixed marker when the passages are insufficient.
- **Citations are numbered passages.** The model cites `[n]`; the API returns only the passages that were cited.
- **Layered code.** Routers handle HTTP only, services hold the logic, repositories hold the queries. Dependencies are injected, which is what makes the services testable with fakes.
- **Swappable providers.** Embeddings and chat models are chosen by configuration, not code.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

The tests use a deterministic fake embedder and fake chat model, so they run offline and do not need a GPU, API keys or Tesseract. They cover cleaning and chunking, the FAISS wrapper, PDF extraction and the OCR fallback, repositories, the ingestion pipeline, index rebuilding, grounded answering, the LangGraph flow and the HTTP API.

## Deployment

**Backend on Render.** Create a Blueprint from this repository (`render.yaml`). It provisions a free PostgreSQL database and a Docker web service that installs Tesseract. Set `HF_API_TOKEN`, `LLM_API_KEY` and `CORS_ORIGINS` (your Vercel URL) in the dashboard. The free instance has limited memory, so the Docker image uses `EMBEDDING_PROVIDER=hf_api` and does not install torch.

**Frontend on Vercel.** Import the repository, set the root directory to `frontend` and add `VITE_API_URL` pointing at the Render service URL.

Free Render services sleep when idle, so the first request after a pause is slow.

## Limitations and next steps

- Retrieval grading is based on similarity scores. An LLM-based relevance grader would be a natural upgrade for the agentic flow.
- Questions that need aggregation across many reports ("what were the most common causes?") are limited by `TOP_K`. Increasing `TOP_K` or adding a map-reduce summarisation step would help.
- No authentication. Add it before exposing the service publicly with sensitive documents.
- Add hybrid search (BM25 plus embeddings) and a reranker to improve precision on exact terms such as site names.
