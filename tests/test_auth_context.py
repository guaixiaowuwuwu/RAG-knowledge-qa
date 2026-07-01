from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.security.auth import get_request_context
from app.security.context import RequestContext


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _settings(**overrides):
    values = {
        "auth_enabled": False,
        "admin_api_keys": "",
        "default_tenant_id": "default",
        "permission_version": "local-v1",
        "max_question_chars": 2000,
        "retrieval_top_k": 4,
        "answer_cache_enabled": False,
        "chat_model": "gpt-4o-mini",
        "embedding_model": "bge-m3",
        "chroma_collection": "test",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_local_mode_keeps_ask_flow_without_credentials(monkeypatch):
    from app.api import routes
    from app.security import auth as auth_module

    captured = {}

    class FakeAnswer:
        answer = "local answer"
        sources = []
        debug = None

    class FakeService:
        def answer(self, **kwargs):
            captured.update(kwargs)
            return FakeAnswer()

    settings = _settings(auth_enabled=False)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "build_rag_service", lambda: FakeService())

    client = TestClient(app)
    response = client.post("/ask", json={"question": "Q", "top_k": 2})

    assert response.status_code == 200
    assert response.json()["answer"] == "local answer"
    assert isinstance(captured["context"], RequestContext)
    assert captured["context"].source == "local"
    assert captured["context"].roles == ("admin",)


def test_auth_enabled_ask_requires_credentials(monkeypatch):
    from app.security import auth as auth_module

    settings = _settings(auth_enabled=True, admin_api_keys="secret")
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    client = TestClient(app)
    response = client.post("/ask", json={"question": "Q", "top_k": 2})

    assert response.status_code == 401


def test_non_admin_cannot_call_ingest(monkeypatch):
    from app.api import routes

    app.dependency_overrides[get_request_context] = lambda: RequestContext(
        tenant_id="default",
        user_id="user-1",
        display_name="User",
        department_ids=("sales",),
        roles=("employee",),
        permission_version="local-v1",
        source="wecom",
    )
    monkeypatch.setattr(routes, "get_settings", lambda: _settings())

    client = TestClient(app)
    response = client.post("/ingest")

    assert response.status_code == 403


def test_admin_api_key_can_call_admin_endpoints(monkeypatch, tmp_path):
    from app.api import routes
    from app.security import auth as auth_module

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    bm25_path = tmp_path / "bm25.jsonl"
    parent_path = tmp_path / "parents.jsonl"
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    settings = _settings(
        auth_enabled=True,
        admin_api_keys="secret",
        documents_dir=documents_dir,
        bm25_corpus_path=bm25_path,
        parent_corpus_path=parent_path,
        chroma_dir=chroma_dir,
        chunk_size=16,
        chunk_overlap=4,
        parent_chunk_size=32,
        parent_chunk_overlap=8,
    )
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "build_vector_store", lambda: object())
    monkeypatch.setattr(
        routes,
        "ingest_directory",
        lambda **kwargs: SimpleNamespace(
            loaded_documents=1,
            indexed_chunks=2,
            skipped=[],
            errors={},
        ),
    )

    client = TestClient(app)
    response = client.post("/ingest", headers={"x-admin-api-key": "secret"})

    assert response.status_code == 200
    assert response.json()["indexed_chunks"] == 2
