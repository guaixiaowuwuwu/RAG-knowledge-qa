import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from app.api.routes import build_rag_service
from app.core.config import get_settings
from app.evaluation.answer_eval import (
    RagasEvaluationError,
    build_answer_eval_record,
    run_ragas_evaluation,
    summarize_answer_eval,
)
from app.evaluation.dataset import load_eval_cases
from scripts.evaluate import config_snapshot, default_output_path, write_json_report


MIXED_SAMPLE_CATEGORIES = [
    "exact_fact",
    "summary",
    "policy",
    "sec_filing",
    "comparison",
    "risk",
    "negative",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run required RAGAS answer-level evaluation.")
    parser.add_argument("--dataset", type=Path, default=None, help="Evaluation JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of cases.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Output JSON report path.")
    parser.add_argument("--sample", choices=["first", "mixed"], default="first", help="Case sampling strategy.")
    parser.add_argument(
        "--include-categories",
        default="",
        help="Comma-separated categories to include, e.g. exact_fact,summary,policy,negative.",
    )
    parser.add_argument("--language", choices=["zh", "en", "all"], default="all")
    parser.add_argument("--random-seed", type=int, default=None, help="Optional seed for randomized sampling order.")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS and save local diagnostics only.")
    return parser


def select_answer_eval_cases(
    cases,
    *,
    limit: int | None,
    sample: str,
    include_categories: list[str] | None = None,
    language: str = "all",
    random_seed: int | None = None,
):
    include_categories = include_categories or []
    filtered = [
        case
        for case in cases
        if (not include_categories or case.category in include_categories)
        and (language == "all" or case.language == language)
    ]

    if sample == "mixed":
        selected = _mixed_sample(filtered, include_categories or MIXED_SAMPLE_CATEGORIES, limit, random_seed)
    else:
        selected = _ordered_or_seeded(filtered, random_seed)
        if limit is not None:
            selected = selected[:limit]

    return selected


def parse_categories(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _mixed_sample(cases, categories: list[str], limit: int | None, random_seed: int | None):
    by_category = {category: [] for category in categories}
    for case in cases:
        if case.category in by_category:
            by_category[case.category].append(case)

    if random_seed is not None:
        rng = random.Random(random_seed)
        for rows in by_category.values():
            rng.shuffle(rows)

    selected = []
    target = limit or len(cases)
    while len(selected) < target:
        added = False
        for category in categories:
            rows = by_category.get(category, [])
            if rows:
                selected.append(rows.pop(0))
                added = True
                if len(selected) >= target:
                    break
        if not added:
            break
    return selected


def _ordered_or_seeded(cases, random_seed: int | None):
    selected = list(cases)
    if random_seed is not None:
        random.Random(random_seed).shuffle(selected)
    return selected


def run_answer_evaluation(cases, service, *, top_k: int, model_config: dict) -> list[dict]:
    records = []
    for case in cases:
        answer = service.answer(case.question, top_k=top_k)
        records.append(build_answer_eval_record(case, answer, model_config))
    return records


def write_failed_report(report: dict, args, reason: str) -> Path:
    output_path = args.output or default_output_path("answer-eval-failed")
    report["error"] = reason
    write_json_report(report, output_path)
    return output_path


def disabled_ragas_payload(reason: str) -> dict:
    return {
        "enabled": False,
        "required": False,
        "reason": reason,
    }


def print_report_summary(report: dict, output_path: Path) -> None:
    print(json.dumps({"local_summary": report["summary"]}, ensure_ascii=False, indent=2))
    print(json.dumps({"ragas": report.get("ragas")}, ensure_ascii=False, indent=2))
    print(f"Saved report: {output_path}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args([] if argv is None else argv)
    settings = get_settings()
    dataset_path = args.dataset or settings.eval_dataset_path
    top_k = args.top_k or settings.retrieval_top_k
    all_cases = load_eval_cases(dataset_path)
    include_categories = parse_categories(args.include_categories)
    cases = select_answer_eval_cases(
        all_cases,
        limit=args.limit,
        sample=args.sample,
        include_categories=include_categories,
        language=args.language,
        random_seed=args.random_seed,
    )

    model_config = config_snapshot(settings)
    model_config["answer_eval_generated_at"] = datetime.now().isoformat()
    report = {
        "dataset_path": str(dataset_path),
        "top_k": top_k,
        "selection": {
            "sample": args.sample,
            "limit": args.limit,
            "include_categories": include_categories,
            "language": args.language,
            "random_seed": args.random_seed,
            "selected_cases": len(cases),
        },
        "config": model_config,
        "summary": summarize_answer_eval([]),
        "records": [],
    }

    try:
        service = build_rag_service()
        records = run_answer_evaluation(cases, service, top_k=top_k, model_config=model_config)
    except Exception as exc:
        reason = f"Answer generation failed: {type(exc).__name__}: {exc}"
        output_path = write_failed_report(report, args, reason)
        raise SystemExit(f"{reason}. Partial report saved: {output_path}") from exc

    report["summary"] = summarize_answer_eval(records)
    report["records"] = records

    if args.no_ragas:
        report["ragas"] = disabled_ragas_payload("--no-ragas was provided")
    else:
        try:
            report["ragas"] = run_ragas_evaluation(
                records,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                judge_model=settings.chat_model,
                embedding_model="bge-m3",
            )
        except RagasEvaluationError as exc:
            output_path = write_failed_report(report, args, f"RAGAS answer evaluation failed: {exc}")
            raise SystemExit(f"RAGAS answer evaluation failed: {exc}. Partial report saved: {output_path}") from exc

    output_path = args.output or default_output_path("answer-eval")
    write_json_report(report, output_path)
    print_report_summary(report, output_path)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
