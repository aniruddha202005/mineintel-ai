import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import create_app
from tests.fakes import FakeEmbedder, fake_llm, make_pdf_bytes

ROOF_FALL = "A roof fall trapped two miners in the east stope after loose strata were not scaled before the shift."


@pytest.fixture
def client(isolated_app_state):
    app = create_app()
    app.dependency_overrides[deps.get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[deps.get_llm] = lambda: fake_llm("Loose strata were not scaled [1].")
    with TestClient(app) as test_client:
        yield test_client


def upload(client, content=None, name="report.pdf"):
    content = content or make_pdf_bytes([ROOF_FALL])
    return client.post("/api/documents", files={"file": (name, content, "application/pdf")})


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "indexed_chunks": 0}


def test_upload_list_and_delete_document(client):
    created = upload(client)
    assert created.status_code == 201
    body = created.json()
    assert body["filename"] == "report.pdf"
    assert body["chunk_count"] == 1

    assert [d["id"] for d in client.get("/api/documents").json()] == [body["id"]]
    assert client.get("/api/health").json()["indexed_chunks"] == 1

    assert client.delete(f"/api/documents/{body['id']}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{body['id']}").status_code == 404


def test_duplicate_and_invalid_uploads(client):
    content = make_pdf_bytes([ROOF_FALL])
    assert upload(client, content).status_code == 201
    assert upload(client, content).status_code == 409
    assert upload(client, b"plain text", "notes.txt").status_code == 422


def test_question_returns_cited_sources(client):
    upload(client)
    response = client.post("/api/query", json={"question": "What caused the roof fall in the east stope?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["sources"][0]["document_name"] == "report.pdf"
    assert body["sources"][0]["page_number"] == 1
    assert "roof fall" in body["sources"][0]["text"]


def test_question_without_documents_is_not_grounded(client):
    body = client.post("/api/query", json={"question": "What caused the roof fall?"}).json()
    assert body["grounded"] is False
    assert body["sources"] == []


def test_query_history_is_recorded(client):
    upload(client)
    client.post("/api/query", json={"question": "What caused the roof fall in the east stope?", "agentic": True})
    history = client.get("/api/query/history").json()
    assert len(history) == 1
    assert history[0]["agentic"] is True


def test_short_questions_are_rejected(client):
    assert client.post("/api/query", json={"question": "hi"}).status_code == 422
