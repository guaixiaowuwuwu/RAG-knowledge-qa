from pathlib import Path
from types import SimpleNamespace

from app.api import routes
from app.rag.hybrid_retriever import HybridRetriever


def test_build_retriever_uses_required_bge_reranker(monkeypatch):
    created = {}
    settings = SimpleNamespace(
        chroma_dir=Path("data/chroma"),
        chroma_collection="test",
        embedding_model="bge-m3",
        openai_api_key="test-key",
        openai_base_url="https://example.com/v1",
        chat_model="test-chat",
        bm25_corpus_path=Path("data/chroma/bm25_corpus.jsonl"),
        parent_corpus_path=Path("data/chroma/parent_corpus.jsonl"),
        reranker_model="BAAI/bge-reranker-v2-m3",
        dense_retrieval_top_k=20,
        bm25_retrieval_top_k=20,
        rrf_k=60,
        reranker_top_n=5,
        query_rewrite_enabled=True,
        hyde_enabled=True,
        max_query_variants=4,
        query_transform_timeout_seconds=8.0,
    )

    class FakeDenseRetriever:
        pass

    class FakeSparseRetriever:
        @classmethod
        def from_jsonl(cls, corpus_path):
            created["corpus_path"] = corpus_path
            return cls()

    class FakeReranker:
        pass

    class FakeLLM:
        def __init__(self, api_key: str, base_url: str, model: str, timeout_seconds: float | None = None):
            created["llm"] = (api_key, base_url, model, timeout_seconds)

        def complete(self, prompt: str) -> str:
            return ""

    def fake_build_bge_reranker(model_name: str):
        created["reranker_model"] = model_name
        return FakeReranker()

    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "build_vector_store", lambda: FakeDenseRetriever())
    monkeypatch.setattr(routes, "BM25Retriever", FakeSparseRetriever)
    monkeypatch.setattr(routes, "build_bge_reranker", fake_build_bge_reranker)
    monkeypatch.setattr(routes, "OpenAIChatLLM", FakeLLM)

    retriever = routes.build_retriever()

    assert isinstance(retriever, HybridRetriever)
    assert created["reranker_model"] == "BAAI/bge-reranker-v2-m3"
    assert created["corpus_path"] == Path("data/chroma/bm25_corpus.jsonl")
    assert created["llm"] == ("test-key", "https://example.com/v1", "test-chat", 60.0)
    assert retriever.parent_store.path == Path("data/chroma/parent_corpus.jsonl")
    assert retriever.query_transformer.max_variants == 4
    assert retriever.query_transformer.timeout_seconds == 8.0
