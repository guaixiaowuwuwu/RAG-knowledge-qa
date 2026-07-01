import json
from pathlib import Path
from types import SimpleNamespace

from app.evaluation.dataset import EvalCase, EvalDatasetError, load_eval_cases
from app.evaluation.metrics import hit_rate_at_k, refusal_reason_counts
from app.evaluation.report import build_retrieval_report
from app.ingestion.chunker import Chunk
from app.rag.documents import RetrievedDocument
from scripts import evaluate
from scripts.evaluate import retrieve_cases


def test_load_eval_cases_from_jsonl(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "rag-basic",
                "question": "RAG 是什么？",
                "ground_truth": "检索增强生成",
                "expected_sources": ["rag.md"],
                "expected_answer_keywords": ["检索", "生成"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_eval_cases(path)

    assert cases[0].question == "RAG 是什么？"
    assert cases[0].expected_sources == ["rag.md"]
    assert cases[0].id == "rag-basic"


def test_hit_rate_at_k_counts_expected_source_match():
    retrieved = [
        [RetrievedDocument(id="1", content="x", source="rag.md", metadata={})],
        [RetrievedDocument(id="2", content="y", source="other.md", metadata={})],
    ]
    expected_sources = [["rag.md"], ["missing.md"]]

    assert hit_rate_at_k(retrieved, expected_sources) == 0.5


def test_refusal_reason_counts_skips_non_refusals():
    assert refusal_reason_counts(["time_sensitive", None, "time_sensitive", "empty_retrieval"]) == {
        "time_sensitive": 2,
        "empty_retrieval": 1,
    }


def test_retrieval_report_includes_grouped_summaries():
    cases = [
        {
            "id": "zh-fact-hit",
            "question": "问题 A",
            "ground_truth": "答案 A",
            "expected_sources": ["a.md"],
            "expected_answer_keywords": ["A"],
            "retrieved": [RetrievedDocument(id="1", content="A", source="a.md", metadata={})],
            "category": "exact_fact",
            "difficulty": "easy",
            "language": "zh",
            "is_negative": False,
        },
        {
            "id": "en-risk-miss",
            "question": "Question B",
            "ground_truth": "Answer B",
            "expected_sources": ["b.md"],
            "expected_answer_keywords": ["B"],
            "retrieved": [RetrievedDocument(id="2", content="x", source="x.md", metadata={})],
            "category": "risk",
            "difficulty": "hard",
            "language": "en",
            "is_negative": False,
        },
        {
            "id": "negative",
            "question": "Today stock price?",
            "ground_truth": "Cannot answer.",
            "expected_sources": [],
            "expected_answer_keywords": [],
            "retrieved": [],
            "category": "negative",
            "difficulty": "easy",
            "language": "en",
            "is_negative": True,
            "refusal_reason": "time_sensitive",
        },
    ]

    report = build_retrieval_report(cases)

    assert report["groups"]["language"]["zh"]["hit_rate_at_k"] == 1.0
    assert report["groups"]["language"]["en"]["positive_cases"] == 1
    assert report["groups"]["category"]["risk"]["hit_rate_at_k"] == 0.0
    assert report["groups"]["difficulty"]["easy"]["cases"] == 2
    assert report["groups"]["is_negative"]["true"]["negative_rejection_rate"] == 1.0
    assert report["cases"][0]["language"] == "zh"
    assert report["cases"][1]["category"] == "risk"


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
    cases = [EvalCase(id="rag-basic", question="RAG 是什么？", ground_truth="", expected_sources=["rag.md"])]

    retrieved = retrieve_cases(cases, retriever, top_k=3)

    assert retriever.calls == [("RAG 是什么？", 3)]
    assert retrieved[0][0].source == "rag.md"


def test_build_report_for_variant_applies_confidence_gating_to_negative_cases(monkeypatch):
    class FakeRetriever:
        def similarity_search(self, query: str, top_k: int):
            return [
                Chunk(
                    content="年报披露年度经营数据，但不包含实时股价。",
                    source="report.md",
                    metadata={"source": "report.md", "chunk_index": 0},
                )
            ]

    case = EvalCase(
        id="negative-stock",
        question="比亚迪今天的股票收盘价是多少？",
        ground_truth="知识库不包含实时股价。",
        expected_sources=[],
        is_negative=True,
    )
    monkeypatch.setattr(evaluate, "get_settings", lambda: SimpleNamespace())

    report = evaluate.build_report_for_variant(
        [case],
        FakeRetriever(),
        variant="dense",
        dataset_path=Path("eval.jsonl"),
        top_k=3,
        config={},
    )

    assert report["summary"]["negative_rejection_rate"] == 1.0
    assert report["summary"]["refusal_reasons"] == {"time_sensitive": 1}
    assert report["cases"][0]["retrieved"] == []
    assert report["cases"][0]["refusal_reason"] == "time_sensitive"


def test_config_snapshot_includes_effective_index_paths(tmp_path: Path):
    settings = SimpleNamespace(
        embedding_model="bge-m3",
        chat_model="gpt-4o-mini",
        chroma_collection="test",
        chroma_dir=tmp_path / "legacy" / "chroma",
        bm25_corpus_path=tmp_path / "legacy" / "bm25.jsonl",
        parent_corpus_path=tmp_path / "legacy" / "parents.jsonl",
        index_root_dir=tmp_path / "indexes",
        active_index_version_path=tmp_path / "indexes" / "active_version.txt",
        document_index_version="configured",
        versioned_indexing_enabled=True,
    )

    config = evaluate.config_snapshot(settings)

    assert config["versioned_indexing_enabled"] is True
    assert config["effective_index_version"] == "configured"
    assert config["effective_chroma_dir"] == str(tmp_path / "indexes" / "configured" / "chroma")
    assert config["effective_bm25_corpus_path"] == str(tmp_path / "indexes" / "configured" / "bm25_corpus.jsonl")
    assert config["effective_parent_corpus_path"] == str(tmp_path / "indexes" / "configured" / "parent_corpus.jsonl")
    assert config["bm25_corpus_path"] == str(tmp_path / "legacy" / "bm25.jsonl")


def test_build_retriever_variant_uses_effective_index_paths(monkeypatch, tmp_path: Path):
    captured = {}
    settings = SimpleNamespace(
        chroma_collection="test",
        chroma_dir=tmp_path / "legacy" / "chroma",
        bm25_corpus_path=tmp_path / "legacy" / "bm25.jsonl",
        parent_corpus_path=tmp_path / "legacy" / "parents.jsonl",
        index_root_dir=tmp_path / "indexes",
        active_index_version_path=tmp_path / "indexes" / "active_version.txt",
        document_index_version="configured",
        versioned_indexing_enabled=True,
        reranker_model="BAAI/bge-reranker-v2-m3",
        dense_retrieval_top_k=20,
        bm25_retrieval_top_k=20,
        rrf_k=60,
        reranker_top_n=5,
    )

    class FakeSparseRetriever:
        @classmethod
        def from_jsonl(cls, corpus_path):
            captured["bm25_corpus_path"] = corpus_path
            return cls()

    class FakeParentStore:
        def __init__(self, path):
            captured["parent_corpus_path"] = path

    monkeypatch.setattr(evaluate, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluate, "build_vector_store", lambda: object())
    monkeypatch.setattr(evaluate, "BM25Retriever", FakeSparseRetriever)
    monkeypatch.setattr(evaluate, "JsonlParentStore", FakeParentStore)
    monkeypatch.setattr(evaluate, "build_bge_reranker", lambda model_name: object())

    evaluate.build_retriever_variant("hybrid-rerank-parent")

    assert captured["bm25_corpus_path"] == tmp_path / "indexes" / "configured" / "bm25_corpus.jsonl"
    assert captured["parent_corpus_path"] == tmp_path / "indexes" / "configured" / "parent_corpus.jsonl"


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
    monkeypatch.setattr(
        evaluate,
        "get_settings",
        lambda: SimpleNamespace(
            retrieval_top_k=4,
            chroma_dir=Path("data/chroma"),
            bm25_corpus_path=Path("data/chroma/bm25_corpus.jsonl"),
            parent_corpus_path=Path("data/chroma/parent_corpus.jsonl"),
            document_index_version="local-index-v1",
            versioned_indexing_enabled=False,
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "load_eval_cases",
        lambda path, **kwargs: [EvalCase(id="rag-basic", question="RAG 是什么？", ground_truth="", expected_sources=["rag.md"])],
    )
    monkeypatch.setattr(evaluate, "build_retriever", lambda: retriever)
    monkeypatch.setattr(evaluate, "write_json_report", lambda report, output_path: output_path)

    evaluate.main()

    assert retriever.calls == [("RAG 是什么？", 4)]
    assert '"hit_rate_at_k": 1.0' in capsys.readouterr().out


def test_eval_dataset_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    row = {
        "id": "dup",
        "question": "问题？",
        "ground_truth": "答案",
        "expected_sources": ["rag.md"],
        "expected_answer_keywords": ["答案"],
    }
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for _ in range(2)),
        encoding="utf-8",
    )

    try:
        load_eval_cases(path)
    except EvalDatasetError as exc:
        assert "duplicate id" in str(exc)
    else:
        raise AssertionError("expected duplicate id validation error")


def test_negative_eval_case_can_have_no_expected_sources(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "negative-1",
                "question": "知识库是否包含火星基地预算？",
                "ground_truth": "知识库没有该信息。",
                "expected_sources": [],
                "expected_answer_keywords": ["没有"],
                "is_negative": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    case = load_eval_cases(path)[0]

    assert case.is_negative is True
    assert case.expected_sources == []
