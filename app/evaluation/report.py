from datetime import UTC, datetime

from app.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    negative_rejection_rate,
    precision_at_k,
    source_recall,
)


def build_retrieval_report(
    cases: list[dict],
    *,
    dataset_path: str | None = None,
    variant: str | None = None,
    top_k: int | None = None,
    config: dict | None = None,
) -> dict:
    retrieved = [case["retrieved"] for case in cases]
    expected_sources = [case["expected_sources"] for case in cases]

    case_rows = []
    for case in cases:
        retrieved_sources = [document.source for document in case["retrieved"]]
        expected = set(case["expected_sources"])
        hit = bool(set(retrieved_sources) & expected)
        case_rows.append(
            {
                "id": case.get("id", ""),
                "question": case["question"],
                "ground_truth": case.get("ground_truth", ""),
                "expected_sources": case["expected_sources"],
                "expected_answer_keywords": case.get("expected_answer_keywords", []),
                "retrieved_sources": retrieved_sources,
                "retrieved": [_serialize_document(document) for document in case["retrieved"]],
                "hit": hit,
                "is_negative": case.get("is_negative", False),
            }
        )

    positive_cases = sum(1 for sources in expected_sources if sources)
    negative_cases = len(expected_sources) - positive_cases
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_path": dataset_path,
        "variant": variant,
        "top_k": top_k,
        "config": config or {},
        "summary": {
            "cases": len(cases),
            "positive_cases": positive_cases,
            "negative_cases": negative_cases,
            "hit_rate_at_k": hit_rate_at_k(retrieved, expected_sources),
            "mrr_at_k": mean_reciprocal_rank(retrieved, expected_sources),
            "source_recall": source_recall(retrieved, expected_sources),
            "source_recall_at_k": source_recall(retrieved, expected_sources),
            "precision_at_k": precision_at_k(retrieved, expected_sources),
            "ndcg_at_k": ndcg_at_k(retrieved, expected_sources),
            "negative_rejection_rate": negative_rejection_rate(retrieved, expected_sources),
        },
        "cases": case_rows,
    }


def _serialize_document(document) -> dict:
    metadata = dict(document.metadata)
    return {
        "id": document.id,
        "source": document.source,
        "score": document.score,
        "page": metadata.get("page"),
        "chunk_index": metadata.get("chunk_index"),
        "matched_child_chunk_index": metadata.get("matched_child_chunk_index"),
        "content": document.content,
    }
