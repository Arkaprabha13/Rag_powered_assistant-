# RAG-Powered Multi-Agent Q&A Assistant

## Overview

This project combines Retrieval-Augmented Generation (RAG) with a tool-routing architecture for document-grounded question answering. Queries are routed to specialized RAG, calculator, or dictionary tools, while the Streamlit interface supports document ingestion and interactive answers.

## Technology Stack

- **Retrieval:** FAISS similarity search with top-3 retrieval
- **Embeddings:** Hugging Face `intfloat/e5-base-v2`
- **LLM:** Groq API with Llama 3
- **Document processing:** LangChain, PyMuPDF, RecursiveCharacterTextSplitter
- **UI:** Streamlit
- **API:** FastAPI + Uvicorn
- **Testing:** pytest

## Architecture

1. Documents are loaded from `data/`.
2. PDF files are converted to text.
3. Documents are split into **1,000-character chunks with 200-character overlap**.
4. E5 embeddings are generated and persisted in a FAISS vector store.
5. Incoming queries are routed to the RAG, calculator, or dictionary tool.
6. RAG retrieves the top 3 relevant chunks and uses the retrieved context for LLM generation.
7. The same QA pipeline is exposed through REST endpoints in `api.py`.

## Run the Streamlit application

```bash
python ingest.py
streamlit run app.py
```

## Run the REST API

```bash
uvicorn api:app --reload
```

## Run with Docker

```bash
docker build -t rag-qa-api .
docker run --rm -p 8000:8000 --env-file .env rag-qa-api
```

The FastAPI service is exposed on port `8000`.

Endpoints:

- `GET /health` - service health check
- `POST /query` - process a document-grounded query
- `POST /upload` - upload and ingest TXT/PDF documents

Example:

```json
POST /query
{
  "query": "What are the negative impacts of uncontrolled EV charging?"
}
```

## Testing

Run the API test suite with:

```bash
pytest -q
```

The tests cover the health endpoint, query validation, mocked agent responses, upload validation, and TXT ingestion.

## Retrieval evaluation

A 25-question evaluation set is included under `evaluation/questions.json`. Run:

```bash
python evaluation/run_evaluation.py
```

The runner reports retrieval hit rate by checking whether the expected source document appears in the retrieved context. This keeps the reported metric reproducible rather than claiming an unverified score. For answer-quality evaluation, review the generated answers against the same question set using a 0-2 correctness rubric before publishing a final quality percentage.

## Design Choices

- **1,000-character chunks + 200-character overlap:** balances context preservation with retrieval precision.
- **E5 embeddings:** dense semantic retrieval without requiring an external embedding API.
- **FAISS:** local vector index for fast similarity search.
- **FastAPI wrapper:** exposes the existing QA pipeline for programmatic use without changing the Streamlit workflow.
- **Automated tests:** protects the REST layer from regressions while mocking the external LLM dependency.

## Environment

Create a `.env` file containing:

```
GROQ_API_KEY=your_groq_api_key_here
```
