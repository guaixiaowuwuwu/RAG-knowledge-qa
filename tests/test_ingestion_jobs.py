from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.ingestion.index_versions import activate_index_version, get_active_index_version, get_index_paths
from app.ingestion.jobs import IngestionJobRepository, run_ingestion_job
from app.main import app
from app.security.auth import get_request_context
from app.security.context import RequestContext


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _settings(tmp_path: Path, **overrides):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir(exist_ok=True)
    values = {
        "audit_db_path": tmp_path / "runtime.sqlite3",
        "index_root_dir": tmp_path / "indexes",
        "active_index_version_path": tmp_path / "indexes" / "active_version.txt",
        "document_index_version": "active-v1",
        "versioned_indexing_enabled": True,
        "documents_dir": documents_dir,
        "documents_manifest_path": tmp_path / "documents_manifest.json",
        "default_tenant_id": "default",
        "chroma_collection": "test_collection",
        "chunk_size": 80,
        "chunk_overlap": 8,
        "parent_chunk_size": 160,
        "parent_chunk_overlap": 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _admin_context():
    return RequestContext(
        tenant_id="default",
        user_id="admin",
        display_name="Admin",
        roles=("admin",),
        permission_version="local-v1",
        source="api_key",
    )


class FakeVectorStore:
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir

    def reset(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def add_chunks(self, chunks):
        return len(chunks)


def test_successful_job_creates_versioned_artifacts_and_activates_index(tmp_path: Path):
    settings = _settings(tmp_path)
    repository = IngestionJobRepository.from_settings(settings)
    job = repository.create_job(
        tenant_id="default",
        requested_by="admin",
        input_path=settings.documents_dir,
        target_index_version="index-v2",
    )
    captured = {}

    def fake_ingest(**kwargs):
        captured.update(kwargs)
        kwargs["vector_store"].reset()
        (kwargs["vector_store"].persist_dir / "chroma.sqlite3").write_text("fake", encoding="utf-8")
        kwargs["bm25_corpus_path"].write_text('{"id":"1"}\n', encoding="utf-8")
        kwargs["parent_corpus_path"].write_text('{"id":"p1"}\n', encoding="utf-8")
        return SimpleNamespace(loaded_documents=1, indexed_chunks=1, skipped=[], errors={})

    result = run_ingestion_job(
        settings,
        job.id,
        ingest_fn=fake_ingest,
        vector_store_factory=lambda paths, _settings: FakeVectorStore(paths.chroma_dir),
    )

    stored = repository.get_job(job.id)
    paths = get_index_paths(settings, version="index-v2")
    assert result is not None
    assert stored is not None
    assert stored.status == "succeeded"
    assert paths.bm25_corpus_path.exists()
    assert paths.parent_corpus_path.exists()
    assert (paths.chroma_dir / "chroma.sqlite3").exists()
    assert captured["documents_dir"] == settings.documents_dir
    assert captured["document_index_version"] == "index-v2"
    assert captured["bm25_corpus_path"] == paths.bm25_corpus_path
    assert get_active_index_version(settings) == "index-v2"


def test_failed_job_does_not_change_active_index(tmp_path: Path):
    settings = _settings(tmp_path)
    (tmp_path / "indexes" / "active-v1").mkdir(parents=True)
    activate_index_version(settings, "active-v1")
    repository = IngestionJobRepository.from_settings(settings)
    job = repository.create_job(
        tenant_id="default",
        requested_by="admin",
        input_path=settings.documents_dir,
        target_index_version="broken-v2",
    )

    def failing_ingest(**kwargs):
        raise RuntimeError("loader failed")

    result = run_ingestion_job(
        settings,
        job.id,
        ingest_fn=failing_ingest,
        vector_store_factory=lambda paths, _settings: FakeVectorStore(paths.chroma_dir),
    )

    stored = repository.get_job(job.id)
    assert result is not None
    assert stored is not None
    assert stored.status == "failed"
    assert "loader failed" in (stored.error or "")
    assert get_active_index_version(settings) == "active-v1"


def test_admin_job_and_index_apis(tmp_path: Path, monkeypatch):
    from app.api import routes

    settings = _settings(tmp_path)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = _admin_context

    client = TestClient(app)
    create_response = client.post(
        "/admin/ingestion/jobs",
        json={"input_path": str(settings.documents_dir), "target_index_version": "index-v2"},
    )

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["status"] == "queued"
    assert created["target_index_version"] == "index-v2"

    detail_response = client.get(f"/admin/ingestion/jobs/{created['id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == created["id"]

    paths = get_index_paths(settings, version="index-v2")
    paths.chroma_dir.mkdir(parents=True)
    (paths.chroma_dir / "chroma.sqlite3").write_text("fake", encoding="utf-8")
    paths.bm25_corpus_path.write_text('{"id":"1"}\n', encoding="utf-8")
    paths.parent_corpus_path.write_text('{"id":"p1"}\n', encoding="utf-8")

    activate_response = client.post("/admin/indexes/index-v2/activate")
    list_response = client.get("/admin/indexes")

    assert activate_response.status_code == 200
    assert activate_response.json()["active_version"] == "index-v2"
    assert list_response.status_code == 200
    assert list_response.json()["active_version"] == "index-v2"
    assert list_response.json()["indexes"][0]["version"] == "index-v2"


def test_legacy_ingest_endpoint_queues_job_in_async_mode(tmp_path: Path, monkeypatch):
    from app.api import routes

    settings = _settings(tmp_path, ingestion_mode="async")
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = _admin_context

    response = TestClient(app).post("/ingest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["job_id"]
    assert payload["loaded_documents"] == 0
    repository = IngestionJobRepository.from_settings(settings)
    assert repository.get_job(payload["job_id"]) is not None
