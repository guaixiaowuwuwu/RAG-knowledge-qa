import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.audit.repository import SqliteAuditRepository, hash_identifier
from app.ingestion.chunker import Chunk
from app.ingestion.index_versions import activate_index_version, get_active_index_version, get_index_paths
from app.ingestion.jobs import run_ingestion_job
from app.integrations.wecom.config import WeComSettings
from app.integrations.wecom.handlers import WeComMessageHandler
from app.integrations.wecom.schemas import WeComIncomingMessage, WeComUserMapping
from app.main import app
from app.rag.cache import build_cache_key
from app.rag.documents import RetrievedDocument
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.service import RagService
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
        "audit_question_max_chars": 200,
        "audit_answer_max_chars": 300,
        "auth_enabled": True,
        "max_question_chars": 2000,
        "answer_cache_enabled": False,
        "retrieval_top_k": 2,
        "chat_model": "gpt-4o-mini",
        "embedding_model": "bge-m3",
        "chroma_collection": "enterprise_smoke",
        "document_index_version": "index-v1",
        "versioned_indexing_enabled": True,
        "index_root_dir": tmp_path / "indexes",
        "active_index_version_path": tmp_path / "indexes" / "active_version.txt",
        "documents_dir": documents_dir,
        "documents_manifest_path": tmp_path / "documents_manifest.json",
        "default_tenant_id": "default",
        "permission_version": "perm-v1",
        "chunk_size": 80,
        "chunk_overlap": 8,
        "parent_chunk_size": 160,
        "parent_chunk_overlap": 16,
        "dense_retrieval_top_k": 3,
        "bm25_retrieval_top_k": 3,
        "rrf_k": 60,
        "reranker_top_n": 3,
        "permission_filter_overfetch_max": 9,
        "min_final_source_count": 1,
        "min_reranker_score": -5.0,
        "enable_low_confidence_refusal": True,
        "time_sensitive_refusal_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _context(
    user_id: str,
    *,
    department_ids=(),
    roles=("employee",),
    source="wecom",
):
    return RequestContext(
        tenant_id="default",
        user_id=user_id,
        display_name=user_id.title(),
        department_ids=tuple(department_ids),
        roles=tuple(roles),
        permission_version="perm-v1",
        source=source,
    )


ALICE = _context("alice", department_ids=("finance",))
BOB = _context("bob", department_ids=("hr",))
ADMIN = _context("admin", roles=("admin",), source="api_key")


def _acl_metadata(*, departments=(), public=False, document_version="doc-v1"):
    return {
        "tenant_id": "default",
        "allowed_user_ids": "[]",
        "allowed_department_ids": json.dumps(list(departments), ensure_ascii=False),
        "allowed_roles": "[]",
        "is_public": public,
        "document_version": document_version,
    }


def _enterprise_chunks():
    return [
        Chunk(
            content="财务报销规则：finance-only budget approvals require CFO review.",
            source="finance-policy.md",
            metadata={
                "source": "finance-policy.md",
                "chunk_index": 0,
                "chunk_id": "finance-1",
                **_acl_metadata(departments=("finance",)),
            },
        ),
        Chunk(
            content="人力资源假期规则：HR onboarding and leave policy.",
            source="hr-policy.md",
            metadata={
                "source": "hr-policy.md",
                "chunk_index": 0,
                "chunk_id": "hr-1",
                **_acl_metadata(departments=("hr",)),
            },
        ),
        Chunk(
            content="公共政策：all employees must follow the information security policy.",
            source="public-policy.md",
            metadata={
                "source": "public-policy.md",
                "chunk_index": 0,
                "chunk_id": "public-1",
                **_acl_metadata(public=True),
            },
        ),
    ]


class KeywordDenseRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search(self, query: str, top_k: int, access_filter=None):
        return _rank_chunks(query, self.chunks)[:top_k]


class KeywordSparseRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query: str, top_k: int, access_filter=None):
        documents = [RetrievedDocument(id=chunk.metadata["chunk_id"], content=chunk.content, source=chunk.source, metadata=dict(chunk.metadata), score=1.0) for chunk in _rank_chunks(query, self.chunks)]
        if access_filter is not None:
            documents = [document for document in documents if access_filter.can_access_metadata(document.metadata)]
        return documents[:top_k]


class IdentityReranker:
    def rerank(self, query, documents, top_n):
        return [
            RetrievedDocument(
                id=document.id,
                content=document.content,
                source=document.source,
                metadata=dict(document.metadata),
                score=1.0,
            )
            for document in documents[:top_n]
        ]


class SourceAwareLLM:
    def complete(self, prompt: str) -> str:
        if "finance-policy.md" in prompt:
            return "财务规则要求 CFO review。"
        if "public-policy.md" in prompt:
            return "公共政策要求遵守信息安全制度。"
        if "hr-policy.md" in prompt:
            return "HR 政策涵盖入职和休假。"
        return "知识库中没有找到可支持该问题的资料，无法基于现有资料回答。"


class FakeVectorStore:
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir

    def reset(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def add_chunks(self, chunks):
        return len(chunks)


class MappingStore:
    def get_by_wecom_userid(self, wecom_userid):
        if wecom_userid != "alice":
            return None
        return WeComUserMapping(
            tenant_id="default",
            wecom_userid="alice",
            system_user_id="alice",
            display_name="Alice",
            department_ids=("finance",),
            roles=("employee",),
            permission_version="perm-v1",
        )


def test_enterprise_smoke_scenario(monkeypatch, tmp_path):
    from app.api import routes

    settings = _settings(tmp_path)
    _create_index_artifact(settings, "index-v1")
    _create_index_artifact(settings, "index-v0")
    activate_index_version(settings, "index-v1")
    repository = SqliteAuditRepository.from_settings(settings)
    current_context = {"value": ALICE}

    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app.dependency_overrides[get_request_context] = lambda: current_context["value"]
    monkeypatch.setattr(
        routes,
        "build_rag_service",
        lambda: _build_service(settings, repository),
    )

    client = TestClient(app)

    current_context["value"] = ALICE
    alice_finance = client.post("/ask", json={"question": "财务报销规则是什么？", "top_k": 1})
    assert alice_finance.status_code == 200
    assert "finance-policy.md" in _sources(alice_finance)
    assert "hr-policy.md" not in _sources(alice_finance)

    current_context["value"] = BOB
    bob_finance = client.post("/ask", json={"question": "财务报销规则是什么？", "top_k": 1})
    assert bob_finance.status_code == 200
    assert "finance-policy.md" not in _sources(bob_finance)
    assert _sources(bob_finance) in ([], ["public-policy.md"])

    alice_public, bob_public = _ask_public_policy(client, current_context)
    assert _sources(alice_public) == ["public-policy.md"]
    assert _sources(bob_public) == ["public-policy.md"]

    handler = WeComMessageHandler(
        settings=WeComSettings(enabled=True, response_mode="passive"),
        rag_service=_build_service(settings, repository),
        user_mapping_store=MappingStore(),
    )
    wecom_result = handler.handle_text_message(
        WeComIncomingMessage(
            to_user_name="agent",
            from_user_name="alice",
            create_time=1,
            msg_type="text",
            content="财务报销规则是什么？",
        )
    )
    assert wecom_result.context.user_id == "alice"
    assert wecom_result.context.department_ids == ("finance",)
    assert wecom_result.answer is not None
    assert "finance-policy.md" in [source.source for source in wecom_result.answer.sources]
    assert "hr-policy.md" not in [source.source for source in wecom_result.answer.sources]
    assert "finance-policy.md" in wecom_result.reply.content

    current_context["value"] = ADMIN
    create_job = client.post(
        "/admin/ingestion/jobs",
        json={"input_path": str(settings.documents_dir), "target_index_version": "index-v2"},
    )
    assert create_job.status_code == 200
    job_id = create_job.json()["id"]
    assert create_job.json()["status"] == "queued"
    assert client.get("/admin/audit/sessions").status_code == 200

    current_context["value"] = BOB
    assert client.post("/admin/ingestion/jobs", json={"target_index_version": "blocked"}).status_code == 403
    assert client.get("/admin/audit/sessions").status_code == 403

    current_context["value"] = ADMIN
    audit_payload = client.get("/admin/audit/sessions").json()
    sessions = audit_payload["sessions"]
    assert len(sessions) >= 5
    assert {session["source"] for session in sessions} >= {"wecom"}
    assert {session["index_version"] for session in sessions} == {"index-v1"}
    assert all(session["tenant_id"] == "default" for session in sessions)
    assert all(session["user_id_hash"] for session in sessions)
    assert any(session["user_id_hash"] == hash_identifier("alice") for session in sessions)

    v1_key = build_cache_key("财务报销规则是什么？", top_k=2, settings=settings, context=ALICE)
    run_ingestion_job(
        settings,
        job_id,
        ingest_fn=_fake_ingest,
        vector_store_factory=lambda paths, _settings: FakeVectorStore(paths.chroma_dir),
    )
    assert get_active_index_version(settings) == "index-v2"
    v2_key = build_cache_key("财务报销规则是什么？", top_k=2, settings=settings, context=ALICE)
    assert v1_key != v2_key

    rollback = client.post("/admin/indexes/index-v1/activate")
    assert rollback.status_code == 200
    assert rollback.json()["active_version"] == "index-v1"
    versions = client.get("/admin/indexes").json()
    assert {item["version"] for item in versions["indexes"]} >= {"index-v0", "index-v1", "index-v2"}
    assert any(item["version"] == "index-v1" and item["is_active"] for item in versions["indexes"])


def _build_service(settings, repository):
    chunks = _enterprise_chunks()
    retriever = HybridRetriever(
        dense_retriever=KeywordDenseRetriever(chunks),
        sparse_retriever=KeywordSparseRetriever(chunks),
        reranker=IdentityReranker(),
        dense_top_k=3,
        sparse_top_k=3,
        rrf_k=60,
        reranker_top_n=3,
        permission_filter_overfetch_max=9,
    )
    return RagService(
        retriever=retriever,
        llm=SourceAwareLLM(),
        audit_repository=repository,
    )


def _rank_chunks(query: str, chunks):
    normalized = query.lower()
    if "财务" in normalized or "finance" in normalized:
        preferred = ["finance-policy.md", "public-policy.md", "hr-policy.md"]
    elif "公共" in normalized or "public" in normalized:
        preferred = ["public-policy.md", "finance-policy.md", "hr-policy.md"]
    elif "hr" in normalized or "人力" in normalized:
        preferred = ["hr-policy.md", "public-policy.md", "finance-policy.md"]
    else:
        preferred = ["public-policy.md", "finance-policy.md", "hr-policy.md"]
    rank = {source: index for index, source in enumerate(preferred)}
    return sorted(chunks, key=lambda chunk: rank.get(chunk.source, len(preferred)))


def _ask_public_policy(client, current_context):
    current_context["value"] = ALICE
    alice_public = client.post("/ask", json={"question": "公共政策是什么？", "top_k": 1})
    assert alice_public.status_code == 200
    current_context["value"] = BOB
    bob_public = client.post("/ask", json={"question": "公共政策是什么？", "top_k": 1})
    assert bob_public.status_code == 200
    return alice_public, bob_public


def _sources(response):
    return [source["source"] for source in response.json()["sources"]]


def _create_index_artifact(settings, version: str):
    paths = get_index_paths(settings, version=version)
    paths.chroma_dir.mkdir(parents=True, exist_ok=True)
    (paths.chroma_dir / "chroma.sqlite3").write_text("fake", encoding="utf-8")
    paths.bm25_corpus_path.write_text(json.dumps({"id": version}) + "\n", encoding="utf-8")
    paths.parent_corpus_path.write_text(json.dumps({"id": f"parent-{version}"}) + "\n", encoding="utf-8")


def _fake_ingest(**kwargs):
    paths = SimpleNamespace(
        chroma_dir=kwargs["vector_store"].persist_dir,
        bm25_corpus_path=kwargs["bm25_corpus_path"],
        parent_corpus_path=kwargs["parent_corpus_path"],
    )
    kwargs["vector_store"].reset()
    (paths.chroma_dir / "chroma.sqlite3").write_text("fake", encoding="utf-8")
    paths.bm25_corpus_path.write_text('{"id":"1"}\n', encoding="utf-8")
    paths.parent_corpus_path.write_text('{"id":"p1"}\n', encoding="utf-8")
    return SimpleNamespace(loaded_documents=1, indexed_chunks=1, skipped=[], errors={})
