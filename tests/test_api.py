from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_agent_result():
    fake_agent = MagicMock()
    fake_agent.process_query.return_value = {
        "answer": "Grounded answer",
        "tool_used": "RAG",
        "context": ["relevant context"],
    }

    with patch.object(api, "_agent", fake_agent):
        response = client.post("/query", json={"query": "What is RAG?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer"
    fake_agent.process_query.assert_called_once_with("What is RAG?")


def test_query_rejects_empty_input():
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 400


def test_upload_rejects_unsupported_type():
    response = client.post(
        "/upload",
        files={"file": ("notes.exe", b"not supported", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_txt_ingests_document():
    with patch.object(api, "ingest_docs") as mock_ingest:
        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"retrieval test document", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "uploaded_and_ingested"
    mock_ingest.assert_called_once()
