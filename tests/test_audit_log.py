from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.audit.repository import SqliteAuditRepository
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


def _settings(tmp_path, **overrides):
    values = {
        "audit_db_path": tmp_path / "audit.sqlite3",
        "audit_question_max_chars": 120,
        "audit_answer_max_chars": 160,
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


def _context(*, tenant_id="tenant-a", user_id="user-1", roles=("employee",), source="wecom"):
    return RequestContext(
        tenant_id=tenant_id,
        user_id=user_id,
        display_name=user_id,
        department_ids=("finance",),
        roles=roles,
        permission_version="perm-v1",
        source=source,
    )


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search(self, query: str, top_k: int, context=None):
        return self.chunks[:top_k]


class FakeStreamingLLM:
    def complete(self, prompt: str) -> str:
        return "完整回答包含制度要点。"

    def stream(self, prompt: str):
        yield "第一段"
        yield "第二段"


def test_successful_answer_writes_audit_session_and_sources(monkeypatch, tmp_path):
    from app.api import routes

    settings = _settings(tmp_path)
    context = _context()
    chunk = Chunk(
        content="财务制度要求双人审批。",
        source="finance.md",
        metadata={
            "source": "finance.md",
            "page": 3,
            "chunk_id": "chunk-1",
            "chunk_index": 7,
            "document_version": "doc-v2",
        },
    )

    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: context
    monkeypatch.setattr(
        routes,
        "build_rag_service",
        lambda: RagService(
            retriever=FakeRetriever([chunk]),
            llm=FakeStreamingLLM(),
            audit_repository=SqliteAuditRepository.from_settings(settings),
        ),
    )

    response = TestClient(app).post("/ask", json={"question": "财务制度是什么？", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]

    repository = SqliteAuditRepository.from_settings(settings)
    detail = repository.get_session(payload["session_id"], tenant_id="tenant-a")
    assert detail is not None
    assert detail.session.tenant_id == "tenant-a"
    assert detail.session.user_id_hash != "user-1"
    assert detail.session.redacted_question == "财务制度是什么？"
    assert detail.session.answer_summary == "完整回答包含制度要点。"
    assert detail.session.refusal_reason is None
    assert detail.session.source == "wecom"
    assert detail.sources[0].source_path == "finance.md"
    assert detail.sources[0].page == 3
    assert detail.sources[0].chunk_id == "chunk-1"
    assert detail.sources[0].document_version == "doc-v2"


def test_refusal_writes_refusal_reason_and_empty_sources(monkeypatch, tmp_path):
    from app.api import routes

    settings = _settings(tmp_path)
    context = _context()
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: context
    monkeypatch.setattr(
        routes,
        "build_rag_service",
        lambda: RagService(
            retriever=FakeRetriever([]),
            llm=FakeStreamingLLM(),
            audit_repository=SqliteAuditRepository.from_settings(settings),
        ),
    )

    response = TestClient(app).post("/ask", json={"question": "没有资料的问题", "top_k": 2})

    assert response.status_code == 200
    repository = SqliteAuditRepository.from_settings(settings)
    detail = repository.get_session(response.json()["session_id"], tenant_id="tenant-a")
    assert detail is not None
    assert detail.session.refusal_reason == "empty_retrieval"
    assert detail.sources == []


def test_streaming_completion_writes_one_audit_session(tmp_path):
    settings = _settings(tmp_path)
    repository = SqliteAuditRepository.from_settings(settings)
    service = RagService(
        retriever=FakeRetriever(
            [
                Chunk(
                    content="RAG 支持流式回答。",
                    source="stream.md",
                    metadata={"source": "stream.md", "chunk_id": "stream-1"},
                )
            ]
        ),
        llm=FakeStreamingLLM(),
        audit_repository=repository,
    )

    events = list(service.answer_stream("怎么流式输出？", top_k=4, context=_context()))

    assert [event["event"] for event in events[:3]] == ["token", "token", "sources"]
    sessions = repository.list_sessions(tenant_id="tenant-a")
    assert len(sessions) == 1
    assert sessions[0].answer_summary == "第一段第二段"
    detail = repository.get_session(sessions[0].id, tenant_id="tenant-a")
    assert detail is not None
    assert len(detail.sources) == 1


def test_non_admin_cannot_query_audit(monkeypatch, tmp_path):
    from app.api import routes

    settings = _settings(tmp_path, auth_enabled=True)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: _context(roles=("employee",))

    response = TestClient(app).get("/admin/audit/sessions")

    assert response.status_code == 403


def test_admin_can_query_same_tenant_audit_sessions(monkeypatch, tmp_path):
    from app.api import routes

    settings = _settings(tmp_path, auth_enabled=True)
    repository = SqliteAuditRepository.from_settings(settings)
    session_id = repository.record_qa_session(
        question="管理员能看到的问题",
        answer="管理员能看到的回答",
        sources=[],
        context=_context(tenant_id="tenant-a"),
        refusal_reason=None,
        latency_ms=12.5,
    )
    repository.record_qa_session(
        question="其他租户问题",
        answer="其他租户回答",
        sources=[],
        context=_context(tenant_id="tenant-b", user_id="user-b"),
        refusal_reason=None,
        latency_ms=1.0,
    )

    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: _context(roles=("admin",))

    client = TestClient(app)
    list_response = client.get("/admin/audit/sessions")
    detail_response = client.get(f"/admin/audit/sessions/{session_id}")

    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()["sessions"]] == [session_id]
    assert detail_response.status_code == 200
    assert detail_response.json()["session"]["id"] == session_id
