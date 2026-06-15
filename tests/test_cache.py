import json
from types import SimpleNamespace

from app.rag.cache import InMemoryAnswerCache, build_cache_key


def test_cache_key_is_stable_and_separates_runtime_options():
    settings = SimpleNamespace(
        chat_model="gpt-4o-mini",
        embedding_model="bge-m3",
        chroma_collection="rag_knowledge_base",
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
