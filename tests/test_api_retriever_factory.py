from pathlib import Path
from types import SimpleNamespace

from app.api import routes
from app.rag.hybrid_retriever import HybridRetriever


def test_build_retriever_uses_optional_reranker_factory(monkeypatch):
    created = {}
    settings = SimpleNamespace(
        chroma_dir=Path("data/chroma"),
        chroma_collection="test",
        embedding_model="bge-m3",
        bm25_corpus_path=Path("data/chroma/bm25_corpus.jsonl"),
        reranker_enabled=False,
        reranker_model="BAAI/bge-reranker-v2-m3",
        dense_retrieval_top_k=20,
        bm25_retrieval_top_k=20,
        rrf_k=60,
        reranker_top_n=5,
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

    def fake_build_reranker(enabled: bool, model_name: str):
        created["reranker_enabled"] = enabled
        created["reranker_model"] = model_name
        return FakeReranker()

    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "build_vector_store", lambda: FakeDenseRetriever())
    monkeypatch.setattr(routes, "BM25Retriever", FakeSparseRetriever)
    monkeypatch.setattr(routes, "build_reranker", fake_build_reranker)

    retriever = routes.build_retriever()

    assert isinstance(retriever, HybridRetriever)
    assert created["reranker_enabled"] is False
    assert created["reranker_model"] == "BAAI/bge-reranker-v2-m3"
    assert created["corpus_path"] == Path("data/chroma/bm25_corpus.jsonl")
