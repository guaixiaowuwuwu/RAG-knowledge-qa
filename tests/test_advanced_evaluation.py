import json
from pathlib import Path

from app.evaluation.dataset import load_eval_cases
from app.evaluation.metrics import (
    evidence_keyword_recall_at_k,
    evidence_strict_hit_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    page_hit_rate_at_k,
    precision_at_k,
    source_recall,
)
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
                "id": "rag-keywords",
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


def test_eval_dataset_loads_optional_evidence_annotations(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "rag-evidence",
                "question": "RAG 是什么？",
                "ground_truth": "检索增强生成",
                "expected_sources": ["rag.md"],
                "expected_answer_keywords": ["检索", "生成"],
                "expected_pages": {"rag.md": [3, 4]},
                "expected_chunk_keywords": ["向量检索", "生成回答"],
                "evidence_notes": "Evidence should come from the retrieval architecture section.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    case = load_eval_cases(path)[0]

    assert case.expected_pages == {"rag.md": [3, 4]}
    assert case.expected_chunk_keywords == ["向量检索", "生成回答"]
    assert case.evidence_notes == "Evidence should come from the retrieval architecture section."


def test_retrieval_metrics_calculate_hit_mrr_and_recall():
    retrieved = [docs("a.md", "b.md"), docs("x.md", "y.md")]
    expected = [["b.md"], ["missing.md", "y.md"]]

    assert hit_rate_at_k(retrieved, expected) == 1.0
    assert mean_reciprocal_rank(retrieved, expected) == 0.5
    assert source_recall(retrieved, expected) == 2 / 3
    assert precision_at_k(retrieved, expected) == 0.5
    assert ndcg_at_k(retrieved, expected) > 0


def test_evidence_level_metrics_use_pages_and_chunk_keywords():
    retrieved = [
        [
            RetrievedDocument(
                id="1",
                content="RAG 使用向量检索并生成回答。",
                source="rag.md",
                metadata={"page": 3},
            )
        ],
        [
            RetrievedDocument(
                id="2",
                content="RAG 使用向量检索。",
                source="rag.md",
                metadata={"page": 2},
            )
        ],
        [
            RetrievedDocument(
                id="3",
                content="unrelated",
                source="risk.md",
                metadata={"page": 9},
            )
        ],
    ]
    expected_sources = [["rag.md"], ["rag.md"], ["risk.md"]]
    expected_pages = [{"rag.md": [3]}, {"rag.md": [3]}, {}]
    expected_chunk_keywords = [["向量检索", "生成回答"], ["向量检索"], ["supply chain"]]

    assert page_hit_rate_at_k(retrieved, expected_sources, expected_pages) == 0.5
    assert evidence_keyword_recall_at_k(retrieved, expected_chunk_keywords) == (1.0 + 1.0 + 0.0) / 3
    assert evidence_strict_hit_at_k(retrieved, expected_sources, expected_pages, expected_chunk_keywords) == 1 / 3


def test_ndcg_counts_each_expected_source_once():
    retrieved = [docs("a.md", "a.md", "a.md")]
    expected = [["a.md"]]

    assert ndcg_at_k(retrieved, expected) == 1.0


def test_build_retrieval_report_returns_case_breakdown():
    cases = [
        {
            "question": "问题 A",
            "expected_sources": ["a.md"],
            "expected_answer_keywords": ["A"],
            "expected_pages": {"a.md": [5]},
            "expected_chunk_keywords": ["A"],
            "evidence_notes": "A should be on page 5.",
            "retrieved": [
                RetrievedDocument(id="1", content="A evidence", source="a.md", metadata={"page": 5}),
            ],
        }
    ]

    report = build_retrieval_report(cases)

    assert report["summary"]["cases"] == 1
    assert report["summary"]["hit_rate_at_k"] == 1.0
    assert report["summary"]["source_recall_at_k"] == 1.0
    assert report["summary"]["precision_at_k"] == 1.0
    assert report["summary"]["page_hit_rate_at_k"] == 1.0
    assert report["summary"]["evidence_keyword_recall_at_k"] == 1.0
    assert report["summary"]["evidence_strict_hit_at_k"] == 1.0
    assert report["cases"][0]["hit"] is True
    assert report["cases"][0]["page_hit"] is True
    assert report["cases"][0]["evidence_keyword_matches"] == ["A"]
    assert report["cases"][0]["evidence_strict_hit"] is True
    assert report["cases"][0]["retrieved"][0]["source"] == "a.md"
