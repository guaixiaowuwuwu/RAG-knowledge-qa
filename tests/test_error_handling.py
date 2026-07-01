from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.ingestion.chunker import Chunk
from app.main import app
from app.rag.service import RagService
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


def _context():
    return RequestContext(
        tenant_id="tenant-a",
        user_id="user-1",
        display_name="User",
        department_ids=("finance",),
        roles=("employee",),
        permission_version="perm-v1",
        source="wecom",
    )


class FakeRetriever:
    def similarity_search(self, query: str, top_k: int, context=None):
        return [
            Chunk(
                content="可用的检索片段。",
                source="allowed.md",
                metadata={"source": "allowed.md", "chunk_index": 0},
            )
        ]


class TimeoutLLM:
    def complete(self, prompt: str) -> str:
        raise TimeoutError("provider stack trace secret-token")


class ExplodingService:
    def answer(self, **kwargs):
        raise RuntimeError("internal stack trace secret-token")


def test_rag_service_returns_safe_unavailable_answer_with_allowed_sources():
    service = RagService(retriever=FakeRetriever(), llm=TimeoutLLM())

    response = service.answer("问题", top_k=1, debug=True, context=_context())

    assert response.answer == "回答服务暂时不可用，请稍后重试。"
    assert response.sources[0].source == "allowed.md"
    assert response.refusal_reason == "llm_unavailable"
    assert response.debug["error"]["code"] == "llm_unavailable"
    assert "secret-token" not in str(response.debug)


def test_llm_timeout_api_response_is_safe(monkeypatch):
    from app.api import routes

    monkeypatch.setattr(routes, "get_settings", lambda: _settings())
    app.dependency_overrides[get_request_context] = _context
    monkeypatch.setattr(routes, "build_rag_service", lambda: RagService(retriever=FakeRetriever(), llm=TimeoutLLM()))

    response = TestClient(app).post("/ask", json={"question": "问题", "top_k": 1, "debug": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "回答服务暂时不可用，请稍后重试。"
    assert payload["sources"][0]["source"] == "allowed.md"
    assert payload["debug"]["error"]["code"] == "llm_unavailable"
    assert "secret-token" not in response.text


def test_unhandled_ask_error_returns_safe_api_error(monkeypatch):
    from app.api import routes

    monkeypatch.setattr(routes, "get_settings", lambda: _settings())
    app.dependency_overrides[get_request_context] = _context
    monkeypatch.setattr(routes, "build_rag_service", lambda: ExplodingService())

    response = TestClient(app).post("/ask", json={"question": "问题", "top_k": 1})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_unavailable"
    assert "secret-token" not in response.text


def test_llm_unavailable_stream_does_not_expose_stack_details():
    service = RagService(retriever=FakeRetriever(), llm=TimeoutLLM())

    events = list(service.answer_stream("问题", top_k=1, debug=True, context=_context()))
    serialized = "\n".join(str(event) for event in events)

    assert events[0] == {"event": "token", "data": "回答服务暂时不可用，请稍后重试。"}
    assert any(event["event"] == "sources" and "allowed.md" in event["data"] for event in events)
    assert "secret-token" not in serialized
