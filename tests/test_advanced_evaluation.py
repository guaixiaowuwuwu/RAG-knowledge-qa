import json
from pathlib import Path

from app.evaluation.dataset import load_eval_cases
from app.evaluation.metrics import hit_rate_at_k, mean_reciprocal_rank, source_recall
from app.evaluation.report import build_retrieval_report
from app.rag.documents import RetrievedDocument


def docs(*sources):
    return [
        RetrievedDocument(id=str(index), content=source, source=source, metadata={})
        for index, source in enumerate(sources)
    ]


def test_eval_dataset_loads_expected_answer_keywords(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
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

    case = load_eval_cases(path)[0]

    assert case.expected_answer_keywords == ["检索", "生成"]


def test_retrieval_metrics_calculate_hit_mrr_and_recall():
    retrieved = [docs("a.md", "b.md"), docs("x.md", "y.md")]
    expected = [["b.md"], ["missing.md", "y.md"]]

    assert hit_rate_at_k(retrieved, expected) == 1.0
    assert mean_reciprocal_rank(retrieved, expected) == 0.5
    assert source_recall(retrieved, expected) == 2 / 3


def test_build_retrieval_report_returns_case_breakdown():
    cases = [
        {
            "question": "问题 A",
            "expected_sources": ["a.md"],
            "retrieved": docs("a.md"),
        }
    ]

    report = build_retrieval_report(cases)

    assert report["summary"]["cases"] == 1
    assert report["summary"]["hit_rate_at_k"] == 1.0
    assert report["cases"][0]["hit"] is True
