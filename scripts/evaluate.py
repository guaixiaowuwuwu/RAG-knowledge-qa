import argparse
import json
from pathlib import Path
from datetime import datetime

from app.api.routes import build_retriever, build_vector_store
from app.core.config import get_settings
from app.evaluation.dataset import EvalCase, load_eval_cases
from app.evaluation.report import build_retrieval_report
from app.rag.bm25 import BM25Retriever
from app.rag.documents import RetrievedDocument, chunk_to_retrieved_document
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.llm import OpenAIChatLLM
from app.rag.parent_store import JsonlParentStore
from app.rag.query_transform import QueryTransformer
from app.rag.reranker import build_bge_reranker


RETRIEVER_VARIANTS = [
    "dense",
    "hybrid",
    "hybrid-rerank",
    "hybrid-rerank-parent",
    "full",
]


class IdentityReranker:
    def rerank(self, query: str, documents: list[RetrievedDocument], top_n: int) -> list[RetrievedDocument]:
        return documents[:top_n]


def retrieve_cases(cases: list[EvalCase], retriever, top_k: int) -> list[list[RetrievedDocument]]:
    retrieved: list[list[RetrievedDocument]] = []
    for case in cases:
        chunks = retriever.similarity_search(case.question, top_k=top_k)
        retrieved.append([chunk_to_retrieved_document(chunk) for chunk in chunks])
    return retrieved


def build_retriever_variant(variant: str):
    settings = get_settings()

    if variant not in RETRIEVER_VARIANTS:
        raise ValueError(f"Unsupported retriever variant: {variant}")

    dense = build_vector_store()
    if variant == "dense":
        return dense

    sparse = BM25Retriever.from_jsonl(settings.bm25_corpus_path)
    reranker = IdentityReranker()
    parent_store = None
    query_transformer = None

    if variant in {"hybrid-rerank", "hybrid-rerank-parent", "full"}:
        reranker = build_bge_reranker(settings.reranker_model)

    if variant in {"hybrid-rerank-parent", "full"}:
        parent_store = JsonlParentStore(settings.parent_corpus_path)

    if variant == "full":
        transformer_llm = OpenAIChatLLM(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.chat_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        query_transformer = QueryTransformer(
            llm=transformer_llm,
            rewrite_enabled=settings.query_rewrite_enabled,
            hyde_enabled=settings.hyde_enabled,
            max_variants=settings.max_query_variants,
        )

    return HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
        dense_top_k=settings.dense_retrieval_top_k,
        sparse_top_k=settings.bm25_retrieval_top_k,
        rrf_k=settings.rrf_k,
        reranker_top_n=settings.reranker_top_n,
        parent_store=parent_store,
        query_transformer=query_transformer,
    )


def build_report_for_variant(
    cases: list[EvalCase],
    retriever,
    *,
    variant: str,
    dataset_path: Path,
    top_k: int,
    config: dict,
) -> dict:
    all_retrieved = retrieve_cases(cases, retriever, top_k=top_k)

    report_cases = []
    for case, retrieved_documents in zip(cases, all_retrieved, strict=False):
        report_cases.append(
            {
                "id": case.id,
                "question": case.question,
                "ground_truth": case.ground_truth,
                "expected_sources": case.expected_sources,
                "expected_answer_keywords": case.expected_answer_keywords,
                "is_negative": case.is_negative,
                "retrieved": retrieved_documents,
            }
        )

    return build_retrieval_report(
        report_cases,
        dataset_path=str(dataset_path),
        variant=variant,
        top_k=top_k,
        config=config,
    )


def config_snapshot(settings) -> dict:
    return {
        "embedding_model": getattr(settings, "embedding_model", ""),
        "chat_model": getattr(settings, "chat_model", ""),
        "ragas_judge_source": "business_qa_openai_config",
        "ragas_embedding_model": "bge-m3",
        "chroma_dir": str(getattr(settings, "chroma_dir", "")),
        "chroma_collection": getattr(settings, "chroma_collection", ""),
        "bm25_corpus_path": str(getattr(settings, "bm25_corpus_path", "")),
        "parent_corpus_path": str(getattr(settings, "parent_corpus_path", "")),
        "dense_retrieval_top_k": getattr(settings, "dense_retrieval_top_k", None),
        "bm25_retrieval_top_k": getattr(settings, "bm25_retrieval_top_k", None),
        "rrf_k": getattr(settings, "rrf_k", None),
        "reranker_model": getattr(settings, "reranker_model", ""),
        "reranker_top_n": getattr(settings, "reranker_top_n", None),
        "parent_chunk_size": getattr(settings, "parent_chunk_size", None),
        "parent_chunk_overlap": getattr(settings, "parent_chunk_overlap", None),
        "query_rewrite_enabled": getattr(settings, "query_rewrite_enabled", None),
        "hyde_enabled": getattr(settings, "hyde_enabled", None),
        "max_query_variants": getattr(settings, "max_query_variants", None),
    }


def default_output_path(prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("reports") / f"{prefix}-{timestamp}.json"


def write_json_report(report: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def comparison_table(reports: list[dict]) -> str:
    rows = [
        "| variant | cases | hit_rate@k | mrr@k | source_recall@k | precision@k | ndcg@k | negative_rejection |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        summary = report["summary"]
        rows.append(
            "| {variant} | {cases} | {hit:.3f} | {mrr:.3f} | {recall:.3f} | {precision:.3f} | {ndcg:.3f} | {negative:.3f} |".format(
                variant=report["variant"],
                cases=summary["cases"],
                hit=summary["hit_rate_at_k"],
                mrr=summary["mrr_at_k"],
                recall=summary["source_recall_at_k"],
                precision=summary["precision_at_k"],
                ndcg=summary["ndcg_at_k"],
                negative=summary["negative_rejection_rate"],
            )
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for RAG benchmark variants.")
    parser.add_argument("--dataset", type=Path, default=None, help="Evaluation JSONL path.")
    parser.add_argument("--variant", choices=RETRIEVER_VARIANTS, default="full")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--compare", action="store_true", help="Run all retriever variants and print a comparison table.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON report path.")
    parser.add_argument("--validate-dataset", action="store_true", help="Only validate the evaluation dataset.")
    parser.add_argument("--check-source-files", action="store_true", help="Verify expected source paths exist.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args([] if argv is None else argv)
    settings = get_settings()
    eval_path = args.dataset or getattr(settings, "eval_dataset_path", Path("data/eval/sample_eval.jsonl"))
    top_k = args.top_k or settings.retrieval_top_k
    cases = load_eval_cases(eval_path, check_source_files=args.check_source_files, source_base_dir=Path.cwd())

    if args.validate_dataset:
        print(f"Dataset valid: {eval_path} ({len(cases)} cases)")
        return

    config = config_snapshot(settings)
    if args.compare:
        reports = []
        for variant in RETRIEVER_VARIANTS:
            retriever = build_retriever_variant(variant)
            reports.append(
                build_report_for_variant(
                    cases,
                    retriever,
                    variant=variant,
                    dataset_path=eval_path,
                    top_k=top_k,
                    config=config,
                )
            )
        comparison = {
            "dataset_path": str(eval_path),
            "top_k": top_k,
            "config": config,
            "variants": reports,
            "table": comparison_table(reports),
        }
        output_path = args.output or default_output_path("retrieval-comparison")
        write_json_report(comparison, output_path)
        print(comparison["table"])
        print(f"Saved report: {output_path}")
        return

    if argv is None:
        retriever = build_retriever()
        variant = "full"
    else:
        retriever = build_retriever_variant(args.variant)
        variant = args.variant
    report = build_report_for_variant(
        cases,
        retriever,
        variant=variant,
        dataset_path=eval_path,
        top_k=top_k,
        config=config,
    )
    output_path = args.output or default_output_path("retrieval")
    write_json_report(report, output_path)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
