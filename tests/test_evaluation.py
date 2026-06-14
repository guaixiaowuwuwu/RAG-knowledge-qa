import json
from pathlib import Path
from types import SimpleNamespace

from app.evaluation.dataset import EvalCase, load_eval_cases
from app.evaluation.metrics import hit_rate_at_k
from app.ingestion.chunker import Chunk
from app.rag.documents import RetrievedDocument
from scripts import evaluate
from scripts.evaluate import retrieve_cases


def test_load_eval_cases_from_jsonl(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "RAG 是什么？",
                "ground_truth": "检索增强生成",
                "expected_sources": ["rag.md"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_eval_cases(path)

    assert cases[0].question == "RAG 是什么？"
    assert cases[0].expected_sources == ["rag.md"]


def test_hit_rate_at_k_counts_expected_source_match():
    retrieved = [
        [RetrievedDocument(id="1", content="x", source="rag.md", metadata={})],
        [RetrievedDocument(id="2", content="y", source="other.md", metadata={})],
    ]
    expected_sources = [["rag.md"], ["missing.md"]]

    assert hit_rate_at_k(retrieved, expected_sources) == 0.5


def test_retrieve_cases_uses_real_retriever_interface():
    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def similarity_search(self, query: str, top_k: int):
            self.calls.append((query, top_k))
            return [
                Chunk(
                    content="RAG 系统包含检索和生成。",
                    source="rag.md",
                    metadata={"source": "rag.md", "chunk_index": 0},
                )
            ]

    retriever = FakeRetriever()
    cases = [EvalCase(question="RAG 是什么？", ground_truth="", expected_sources=["rag.md"])]

    retrieved = retrieve_cases(cases, retriever, top_k=3)

    assert retriever.calls == [("RAG 是什么？", 3)]
    assert retrieved[0][0].source == "rag.md"


def test_evaluate_main_uses_build_retriever(monkeypatch, capsys):
    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def similarity_search(self, query: str, top_k: int):
            self.calls.append((query, top_k))
            return [
                Chunk(
                    content="RAG 系统包含检索和生成。",
                    source="rag.md",
                    metadata={"source": "rag.md", "chunk_index": 0},
                )
            ]

    retriever = FakeRetriever()
    monkeypatch.setattr(evaluate, "get_settings", lambda: SimpleNamespace(retrieval_top_k=4))
    monkeypatch.setattr(
        evaluate,
        "load_eval_cases",
        lambda path: [EvalCase(question="RAG 是什么？", ground_truth="", expected_sources=["rag.md"])],
    )
    monkeypatch.setattr(evaluate, "build_retriever", lambda: retriever)

    evaluate.main()

    assert retriever.calls == [("RAG 是什么？", 4)]
    assert "'hit_rate_at_k': 1.0" in capsys.readouterr().out
