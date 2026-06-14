from scripts import warmup


def test_warmup_builds_retriever_and_reports_reranker(monkeypatch, capsys):
    class FakeReranker:
        pass

    class FakeRetriever:
        reranker = FakeReranker()

        def __init__(self):
            self.calls = []

        def similarity_search(self, query: str, top_k: int):
            self.calls.append((query, top_k))
            return []

    retriever = FakeRetriever()
    monkeypatch.setattr(warmup, "build_retriever", lambda: retriever)

    warmup.main()

    output = capsys.readouterr().out
    assert retriever.calls == [("RAG 系统包含哪些核心步骤？", 1)]
    assert "FakeRetriever" in output
    assert "FakeReranker" in output
    assert "ok" in output
