from pathlib import Path

from app.api.routes import build_retriever
from app.core.config import get_settings
from app.evaluation.dataset import EvalCase, load_eval_cases
from app.evaluation.report import build_retrieval_report
from app.rag.documents import RetrievedDocument, chunk_to_retrieved_document


def retrieve_cases(cases: list[EvalCase], retriever, top_k: int) -> list[list[RetrievedDocument]]:
    retrieved: list[list[RetrievedDocument]] = []
    for case in cases:
        chunks = retriever.similarity_search(case.question, top_k=top_k)
        retrieved.append([chunk_to_retrieved_document(chunk) for chunk in chunks])
    return retrieved


def main() -> None:
    settings = get_settings()
    eval_path = settings.eval_dataset_path if hasattr(settings, "eval_dataset_path") else Path("data/eval/sample_eval.jsonl")
    cases = load_eval_cases(eval_path)
    retriever = build_retriever()
    all_retrieved = retrieve_cases(cases, retriever, top_k=settings.retrieval_top_k)

    report_cases = []
    for case, retrieved_documents in zip(cases, all_retrieved, strict=False):
        report_cases.append(
            {
                "question": case.question,
                "expected_sources": case.expected_sources,
                "retrieved": retrieved_documents,
            }
        )

    print(build_retrieval_report(report_cases))


if __name__ == "__main__":
    main()
