from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.audit.repository import SqliteAuditRepository
from app.main import app
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


def test_feedback_can_be_joined_to_same_tenant_session(monkeypatch, tmp_path):
    from app.api import routes

    settings = _settings(tmp_path)
    repository = SqliteAuditRepository.from_settings(settings)
    session_id = repository.record_qa_session(
        question="制度是什么？",
        answer="制度回答",
        sources=[],
        context=_context(),
        refusal_reason=None,
        latency_ms=10,
    )
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: _context()

    response = TestClient(app).post(
        "/feedback",
        json={
            "session_id": session_id,
            "rating": 1,
            "tags": ["useful", "accurate"],
            "comment": "回答可用",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["feedback_id"]

    detail = repository.get_session(session_id, tenant_id="tenant-a")
    assert detail is not None
    assert detail.feedback[0].rating == 1
    assert detail.feedback[0].tags == ("useful", "accurate")
    assert detail.feedback[0].comment == "回答可用"


def test_feedback_must_belong_to_same_tenant(monkeypatch, tmp_path):
    from app.api import routes

    settings = _settings(tmp_path)
    repository = SqliteAuditRepository.from_settings(settings)
    session_id = repository.record_qa_session(
        question="其他租户问题",
        answer="其他租户回答",
        sources=[],
        context=_context(tenant_id="tenant-b", user_id="user-b"),
        refusal_reason=None,
        latency_ms=10,
    )
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: _context(tenant_id="tenant-a")

    response = TestClient(app).post("/feedback", json={"session_id": session_id, "rating": -1})

    assert response.status_code == 404
    assert repository.get_session(session_id, tenant_id="tenant-b").feedback == []


def test_feedback_rejects_unknown_session(monkeypatch, tmp_path):
    from app.api import routes

    settings = _settings(tmp_path)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: _context()

    response = TestClient(app).post("/feedback", json={"session_id": "missing", "rating": 0})

    assert response.status_code == 404


def test_feedback_requires_valid_rating(monkeypatch, tmp_path):
    from app.api import routes

    settings = _settings(tmp_path)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: _context()

    response = TestClient(app).post("/feedback", json={"session_id": "s", "rating": 3})

    assert response.status_code == 422
