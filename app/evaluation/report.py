from app.evaluation.metrics import hit_rate_at_k, mean_reciprocal_rank, source_recall


def build_retrieval_report(cases: list[dict]) -> dict:
    retrieved = [case["retrieved"] for case in cases]
    expected_sources = [case["expected_sources"] for case in cases]

    case_rows = []
    for case in cases:
        retrieved_sources = [document.source for document in case["retrieved"]]
        expected = set(case["expected_sources"])
        hit = bool(set(retrieved_sources) & expected)
        case_rows.append(
            {
                "question": case["question"],
                "expected_sources": case["expected_sources"],
                "retrieved_sources": retrieved_sources,
                "hit": hit,
            }
        )

    return {
        "summary": {
            "cases": len(cases),
            "hit_rate_at_k": hit_rate_at_k(retrieved, expected_sources),
            "mrr_at_k": mean_reciprocal_rank(retrieved, expected_sources),
            "source_recall": source_recall(retrieved, expected_sources),
        },
        "cases": case_rows,
    }
