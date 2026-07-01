import json
from pathlib import Path
from types import SimpleNamespace

from app.rag.cache import InMemoryAnswerCache, build_cache_key
from app.security.context import RequestContext


def test_cache_key_is_stable_and_separates_runtime_options():
    settings = SimpleNamespace(
        chat_model="gpt-4o-mini",
        embedding_model="bge-m3",
        chroma_collection="rag_knowledge_base",
        min_reranker_score=-5.0,
        min_final_source_count=1,
        enable_low_confidence_refusal=True,
        time_sensitive_refusal_enabled=True,
    )

    base_key = build_cache_key(
        question=" 比亚迪信息披露制度是什么？ ",
        top_k=4,
        settings=settings,
        retrieval_options={"rewrite_enabled": True, "hyde_enabled": True},
    )
    same_key = build_cache_key(
        question="比亚迪信息披露制度是什么？",
        top_k=4,
        settings=settings,
        retrieval_options={"hyde_enabled": True, "rewrite_enabled": True},
    )
    changed_key = build_cache_key(
        question="比亚迪信息披露制度是什么？",
        top_k=5,
        settings=settings,
        retrieval_options={"rewrite_enabled": True, "hyde_enabled": True},
    )

    assert base_key == same_key
    assert changed_key != base_key
    assert base_key.startswith("rag-answer:")


def test_cache_key_separates_confidence_settings():
    base_settings = SimpleNamespace(
        chat_model="gpt-4o-mini",
        embedding_model="bge-m3",
        chroma_collection="rag_knowledge_base",
        min_reranker_score=-5.0,
        min_final_source_count=1,
        enable_low_confidence_refusal=True,
        time_sensitive_refusal_enabled=True,
    )
    stricter_settings = SimpleNamespace(
        chat_model="gpt-4o-mini",
        embedding_model="bge-m3",
        chroma_collection="rag_knowledge_base",
        min_reranker_score=0.0,
        min_final_source_count=1,
        enable_low_confidence_refusal=True,
        time_sensitive_refusal_enabled=True,
    )

    base_key = build_cache_key("Q", top_k=4, settings=base_settings)
    stricter_key = build_cache_key("Q", top_k=4, settings=stricter_settings)

    assert base_key != stricter_key


def test_cache_key_separates_users_permission_versions_and_index_versions():
    settings = SimpleNamespace(
        chat_model="gpt-4o-mini",
        embedding_model="bge-m3",
        chroma_collection="rag_knowledge_base",
        min_reranker_score=-5.0,
        min_final_source_count=1,
        enable_low_confidence_refusal=True,
        time_sensitive_refusal_enabled=True,
        document_index_version="index-v1",
    )
    user_a = RequestContext(
        tenant_id="tenant-a",
        user_id="user-a",
        department_ids=("sales",),
        roles=("employee",),
        permission_version="perm-v1",
        source="wecom",
    )
    user_b = RequestContext(
        tenant_id="tenant-a",
        user_id="user-b",
        department_ids=("sales",),
        roles=("employee",),
        permission_version="perm-v1",
        source="wecom",
    )
    user_a_new_permissions = RequestContext(
        tenant_id="tenant-a",
        user_id="user-a",
        department_ids=("sales",),
        roles=("employee",),
        permission_version="perm-v2",
        source="wecom",
    )

    user_a_key = build_cache_key("Q", top_k=4, settings=settings, context=user_a)
    user_b_key = build_cache_key("Q", top_k=4, settings=settings, context=user_b)
    new_permission_key = build_cache_key("Q", top_k=4, settings=settings, context=user_a_new_permissions)
    new_index_key = build_cache_key(
        "Q",
        top_k=4,
        settings=SimpleNamespace(**{**vars(settings), "document_index_version": "index-v2"}),
        context=user_a,
    )

    assert user_a_key != user_b_key
    assert user_a_key != new_permission_key
    assert user_a_key != new_index_key


def test_cache_key_uses_active_index_version_file(tmp_path: Path):
    active_path = tmp_path / "indexes" / "active_version.txt"
    active_path.parent.mkdir(parents=True)
    active_path.write_text("index-v1\n", encoding="utf-8")
    settings = SimpleNamespace(
        chat_model="gpt-4o-mini",
        embedding_model="bge-m3",
        chroma_collection="rag_knowledge_base",
        min_reranker_score=-5.0,
        min_final_source_count=1,
        enable_low_confidence_refusal=True,
        time_sensitive_refusal_enabled=True,
        document_index_version="configured",
        versioned_indexing_enabled=True,
        index_root_dir=tmp_path / "indexes",
        active_index_version_path=active_path,
    )

    v1_key = build_cache_key("Q", top_k=4, settings=settings)
    active_path.write_text("index-v2\n", encoding="utf-8")
    v2_key = build_cache_key("Q", top_k=4, settings=settings)

    assert v1_key != v2_key


def test_in_memory_cache_round_trip_and_expiry():
    cache = InMemoryAnswerCache(ttl_seconds=60)
    payload = {"answer": "cached", "sources": [{"source": "doc.md"}]}

    cache.set("cache-key", payload)

    assert cache.get("cache-key") == payload
    assert cache.get("missing") is None


def test_in_memory_cache_rejects_expired_entries(monkeypatch):
    cache = InMemoryAnswerCache(ttl_seconds=10)
    now = 1_000.0
    monkeypatch.setattr("app.rag.cache.time.time", lambda: now)
    cache.set("cache-key", {"answer": "cached"})

    monkeypatch.setattr("app.rag.cache.time.time", lambda: now + 11)

    assert cache.get("cache-key") is None
