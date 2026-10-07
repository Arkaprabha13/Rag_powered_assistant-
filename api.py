from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from agent import QAAgent
from ingest import ingest_docs

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="RAG-Powered Q&A API",
    version="1.0.0",
    description="REST API for document-grounded question answering.",
)

_agent: QAAgent | None = None


def get_agent() -> QAAgent:
    global _agent
    if _agent is None:
        _agent = QAAgent()
    return _agent


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    return get_agent().process_query(request.query.strip())


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    allowed = {".txt", ".pdf", ".csv", ".json", ".xlsx", ".xml"}
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Use: {', '.join(sorted(allowed))}",
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "").name
    destination = DATA_DIR / filename
    destination.write_bytes(await file.read())

    # The existing ingestion pipeline natively indexes TXT and PDF files.
    # For tabular/XML uploads, callers can convert them to TXT through the
    # Streamlit interface before ingestion.
    if suffix not in {".txt", ".pdf"}:
        return {
            "status": "uploaded",
            "filename": filename,
            "message": "File uploaded. Convert to TXT through the Streamlit document manager, then ingest.",
        }

    ingest_docs()
    return {
        "status": "uploaded_and_ingested",
        "filename": filename,
        "message": "Document uploaded and added to the vector store.",
    }
