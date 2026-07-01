from datetime import UTC, datetime

from app.evaluation.metrics import (
    case_evidence_keyword_matches,
    case_evidence_keyword_recall,
    case_evidence_strict_hit,
    case_page_hit,
    evidence_keyword_recall_at_k,
    evidence_strict_hit_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    negative_rejection_rate,
    page_hit_rate_at_k,
    precision_at_k,
    refusal_reason_counts,
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
    case_rows = []
    for case in cases:
        retrieved_sources = [document.source for document in case["retrieved"]]
        expected = set(case["expected_sources"])
        hit = bool(set(retrieved_sources) & expected)
        page_hit = case_page_hit(case["retrieved"], case["expected_sources"], case.get("expected_pages", {}))
        keyword_matches, keyword_misses = case_evidence_keyword_matches(
            case["retrieved"],
            case.get("expected_chunk_keywords", []),
        )
        keyword_recall = (
            case_evidence_keyword_recall(case["retrieved"], case.get("expected_chunk_keywords", []))
            if case.get("expected_chunk_keywords")
            else None
        )
        strict_hit = case_evidence_strict_hit(
            case["retrieved"],
            case["expected_sources"],
            case.get("expected_pages", {}),
            case.get("expected_chunk_keywords", []),
        )
        case_rows.append(
            {
                "id": case.get("id", ""),
                "question": case["question"],
                "ground_truth": case.get("ground_truth", ""),
                "expected_sources": case["expected_sources"],
                "expected_answer_keywords": case.get("expected_answer_keywords", []),
                "expected_pages": case.get("expected_pages", {}),
                "expected_chunk_keywords": case.get("expected_chunk_keywords", []),
                "evidence_notes": case.get("evidence_notes", ""),
                "category": case.get("category", ""),
                "difficulty": case.get("difficulty", ""),
                "language": case.get("language", ""),
                "retrieved_sources": retrieved_sources,
                "retrieved": [_serialize_document(document) for document in case["retrieved"]],
                "hit": hit,
                "is_negative": case.get("is_negative", False),
                "page_hit": page_hit,
                "evidence_keyword_matches": keyword_matches,
                "evidence_keyword_misses": keyword_misses,
                "evidence_keyword_recall": keyword_recall,
                "evidence_strict_hit": strict_hit,
                "refusal_reason": case.get("refusal_reason"),
                "confidence": case.get("confidence", {}),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_path": dataset_path,
        "variant": variant,
        "top_k": top_k,
        "config": config or {},
        "summary": _summary_for_cases(cases),
        "groups": build_grouped_summaries(cases),
        "cases": case_rows,
    }


def build_grouped_summaries(cases: list[dict]) -> dict:
    return {
        "language": _group_by(cases, "language"),
        "category": _group_by(cases, "category"),
        "difficulty": _group_by(cases, "difficulty"),
        "is_negative": _group_by(cases, "is_negative", normalize=_bool_label),
    }


def _group_by(cases: list[dict], field: str, normalize=None) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for case in cases:
        raw_value = case.get(field)
        label = normalize(raw_value) if normalize is not None else str(raw_value or "unknown")
        grouped.setdefault(label, []).append(case)
    return {label: _summary_for_cases(group_cases) for label, group_cases in grouped.items()}


def _summary_for_cases(cases: list[dict]) -> dict:
    retrieved = [case["retrieved"] for case in cases]
    expected_sources = [case["expected_sources"] for case in cases]
    expected_pages = [case.get("expected_pages", {}) for case in cases]
    expected_chunk_keywords = [case.get("expected_chunk_keywords", []) for case in cases]
    positive_cases = sum(1 for sources in expected_sources if sources)
    negative_cases = len(expected_sources) - positive_cases
    refusal_reasons = [case.get("refusal_reason") for case in cases]
    return {
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
        "page_hit_rate_at_k": page_hit_rate_at_k(retrieved, expected_sources, expected_pages),
        "evidence_keyword_recall_at_k": evidence_keyword_recall_at_k(retrieved, expected_chunk_keywords),
        "evidence_strict_hit_at_k": evidence_strict_hit_at_k(
            retrieved,
            expected_sources,
            expected_pages,
            expected_chunk_keywords,
        ),
        "refusal_reasons": refusal_reason_counts(refusal_reasons),
    }


def _bool_label(value) -> str:
    return "true" if bool(value) else "false"


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
