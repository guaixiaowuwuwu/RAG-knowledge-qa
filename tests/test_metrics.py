import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.ingestion.chunker import Chunk
from app.main import app
from app.observability.metrics import metrics
from app.rag.confidence import REFUSAL_ANSWER
from app.rag.service import RagService
from app.security.auth import get_request_context
from app.security.context import RequestContext


@pytest.fixture(autouse=True)
def clear_state():
    app.dependency_overrides.clear()
    metrics.reset()
    yield
    app.dependency_overrides.clear()
    metrics.reset()


def _settings(**overrides):
    values = {
        "auth_enabled": False,
        "max_question_chars": 2000,
        "answer_cache_enabled": False,
        "retrieval_top_k": 4,
        "chat_model": "gpt-4o-mini",
        "embedding_model": "bge-m3",
        "chroma_collection": "test",
        "document_index_version": "index-v1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _context(*, user_id="user-1"):
    return RequestContext(
        tenant_id="tenant-a",
        user_id=user_id,
        display_name=user_id,
        department_ids=("finance",),
        roles=("employee",),
        permission_version="perm-v1",
        source="wecom",
    )


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search(self, query: str, top_k: int, context=None):
        return self.chunks[:top_k]


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return "可用回答"


class FailingService:
    def answer(self, **kwargs):
        raise RuntimeError("provider failure")


def test_metrics_endpoint_increments_success_refusal_and_error(monkeypatch):
    from app.api import routes

    settings = _settings()
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = _context
    chunk = Chunk(content="片段", source="doc.md", metadata={"source": "doc.md", "chunk_index": 0})
    client = TestClient(app)

    monkeypatch.setattr(routes, "build_rag_service", lambda: RagService(retriever=FakeRetriever([chunk]), llm=FakeLLM()))
    success = client.post("/ask", json={"question": "正常问题", "top_k": 1})
    assert success.status_code == 200

    monkeypatch.setattr(routes, "build_rag_service", lambda: RagService(retriever=FakeRetriever([]), llm=FakeLLM()))
    refusal = client.post("/ask", json={"question": "没有资料的问题", "top_k": 1})
    assert refusal.status_code == 200
    assert refusal.json()["answer"] == REFUSAL_ANSWER

    monkeypatch.setattr(routes, "build_rag_service", lambda: FailingService())
    error = client.post("/ask", json={"question": "异常问题", "top_k": 1})
    assert error.status_code == 503

    metrics_response = client.get("/metrics")
    body = metrics_response.text
    assert metrics_response.status_code == 200
    assert 'rag_http_requests_total{method="POST",path="/ask",status="200"} 2' in body
    assert 'rag_http_requests_total{method="POST",path="/ask",status="503"} 1' in body
    assert 'rag_http_errors_total{method="POST",path="/ask",status="503"} 1' in body
    assert 'rag_refusals_total{reason="empty_retrieval"} 1' in body
    assert 'rag_errors_total{stage="api"} 1' in body


def test_structured_request_log_contains_context_and_no_secrets(monkeypatch, caplog):
    from app.api import routes
    from app import middleware as middleware_module

    settings = _settings(admin_api_keys="super-secret")
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(middleware_module, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: _context(user_id="alice")
    monkeypatch.setattr(routes, "build_rag_service", lambda: RagService(retriever=FakeRetriever([]), llm=FakeLLM()))

    with caplog.at_level(logging.INFO, logger="app.requests"):
        response = TestClient(app).post(
            "/ask",
            headers={"x-request-id": "req-123", "x-admin-api-key": "super-secret"},
            json={"question": "没有资料的问题", "top_k": 1},
        )

    assert response.status_code == 200
    records = [record for record in caplog.records if record.name == "app.requests"]
    assert records
    structured = records[-1].structured
    assert structured["request_id"] == "req-123"
    assert structured["tenant_id"] == "tenant-a"
    assert structured["user_id_hash"]
    assert structured["path"] == "/ask"
    assert structured["index_version"] == "index-v1"
    serialized = json.dumps(structured, ensure_ascii=False)
    assert "alice" not in serialized
    assert "super-secret" not in serialized
