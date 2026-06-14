import json
from pathlib import Path

from app.evaluation.dataset import load_eval_cases
from app.evaluation.metrics import hit_rate_at_k
from app.rag.documents import RetrievedDocument


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
