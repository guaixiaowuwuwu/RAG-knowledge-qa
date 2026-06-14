import sys
from types import ModuleType

from app.rag.documents import RetrievedDocument
from app.rag.reranker import NoopReranker, ScoreBasedReranker, build_bge_reranker, build_reranker


def test_noop_reranker_keeps_order_and_trims_top_n():
    docs = [
        RetrievedDocument(id="a", content="A", source="a.md", metadata={}),
        RetrievedDocument(id="b", content="B", source="b.md", metadata={}),
    ]

    results = NoopReranker().rerank("query", docs, top_n=1)

    assert [doc.id for doc in results] == ["a"]


def test_score_based_reranker_sorts_by_model_score():
    class FakeModel:
        def compute_score(self, pairs):
            assert pairs == [["query", "A"], ["query", "B"]]
            return [0.1, 0.9]

    docs = [
        RetrievedDocument(id="a", content="A", source="a.md", metadata={}),
        RetrievedDocument(id="b", content="B", source="b.md", metadata={}),
    ]

    reranker = ScoreBasedReranker(FakeModel())
    results = reranker.rerank("query", docs, top_n=2)

    assert [doc.id for doc in results] == ["b", "a"]
    assert results[0].score == 0.9


def test_build_bge_reranker_uses_required_model(monkeypatch):
    created = {}

    class FakeFlagReranker:
        def __init__(self, model_name: str, use_fp16: bool):
            created["model_name"] = model_name
            created["use_fp16"] = use_fp16

        def compute_score(self, pairs):
            return [1.0 for _ in pairs]

    fake_module = ModuleType("FlagEmbedding")
    fake_module.FlagReranker = FakeFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)

    reranker = build_bge_reranker("BAAI/bge-reranker-v2-m3")

    assert isinstance(reranker, ScoreBasedReranker)
    assert created == {"model_name": "BAAI/bge-reranker-v2-m3", "use_fp16": True}


def test_build_reranker_defaults_to_noop_when_disabled():
    reranker = build_reranker(False, "BAAI/bge-reranker-v2-m3")

    assert isinstance(reranker, NoopReranker)
