import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from app.api.routes import build_rag_service
from app.core.config import get_settings
from app.evaluation.dataset import load_eval_cases


def percentile(values: list[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile_rank)))
    return round(ordered[index], 3)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 3),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": round(max(values), 3),
    }


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("reports") / f"latency-benchmark-{timestamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight local latency benchmark for the RAG service.")
    parser.add_argument("--dataset", type=Path, default=None, help="Evaluation JSONL path.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of questions to run.")
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval top_k.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON report path.")
    parser.add_argument("--debug", action="store_true", help="Include retrieval trace and timing fields in responses.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    dataset_path = args.dataset or settings.eval_dataset_path
    top_k = args.top_k or settings.retrieval_top_k
    cases = load_eval_cases(dataset_path)[: args.limit]
    service = build_rag_service()

    rows = []
    for case in cases:
        started_at = time.perf_counter()
        answer = service.answer(
            question=case.question,
            top_k=top_k,
            debug=args.debug,
            retrieval_options=None,
        )
        total_ms = round((time.perf_counter() - started_at) * 1000, 3)
        timings = (answer.debug or {}).get("timings_ms", {}) if args.debug else {}
        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "total_ms": total_ms,
                "retrieval_ms": timings.get("retrieval"),
                "llm_ms": timings.get("llm"),
                "source_count": len(answer.sources),
                "answer_chars": len(answer.answer),
            }
        )

    total_values = [row["total_ms"] for row in rows]
    retrieval_values = [row["retrieval_ms"] for row in rows if row["retrieval_ms"] is not None]
    llm_values = [row["llm_ms"] for row in rows if row["llm_ms"] is not None]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark_scope": "local_laptop_or_container_smoke_benchmark_not_production_qps",
        "dataset_path": str(dataset_path),
        "limit": args.limit,
        "top_k": top_k,
        "config": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "chroma_collection": settings.chroma_collection,
            "answer_cache_enabled": settings.answer_cache_enabled,
            "llm_timeout_seconds": settings.llm_timeout_seconds,
        },
        "summary": {
            "total": summarize(total_values),
            "retrieval": summarize(retrieval_values),
            "llm": summarize(llm_values),
        },
        "cases": rows,
    }

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
