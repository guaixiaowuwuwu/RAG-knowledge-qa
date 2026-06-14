from app.rag.documents import RetrievedDocument
from app.rag.reranker import NoopReranker, ScoreBasedReranker


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
