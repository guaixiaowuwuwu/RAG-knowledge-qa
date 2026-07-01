from dataclasses import dataclass
import os
from statistics import mean

from app.evaluation.dataset import EvalCase
from app.rag.embeddings import LOCAL_EMBEDDING_MODELS
from app.rag.service import Answer


UNKNOWN_MARKERS = [
    "无法",
    "没有找到",
    "未找到",
    "不知道",
    "不确定",
    "not enough",
    "no relevant",
    "cannot answer",
    "not found",
]

RAGAS_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

RAGAS_TARGET_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.80,
}


class RagasEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnswerEvaluation:
    keyword_coverage: float
    citation_present: bool
    answer_non_empty: bool
    unknown_answer: bool
    negative_case_pass: bool | None
    passed: bool
    matched_keywords: list[str]
    missing_keywords: list[str]


def evaluate_answer(case: EvalCase, answer: Answer) -> AnswerEvaluation:
    normalized_answer = answer.answer.lower()
    keywords = case.expected_answer_keywords
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in normalized_answer]
    missing_keywords = [keyword for keyword in keywords if keyword.lower() not in normalized_answer]
    keyword_coverage = len(matched_keywords) / len(keywords) if keywords else 0.0
    citation_present = bool(answer.sources)
    answer_non_empty = bool(answer.answer.strip())
    unknown_answer = is_unknown_answer(answer.answer)

    if case.is_negative:
        negative_case_pass = unknown_answer and not citation_present
        passed = negative_case_pass
    else:
        negative_case_pass = None
        passed = answer_non_empty and citation_present and not unknown_answer and keyword_coverage >= 0.5

    return AnswerEvaluation(
        keyword_coverage=keyword_coverage,
        citation_present=citation_present,
        answer_non_empty=answer_non_empty,
        unknown_answer=unknown_answer,
        negative_case_pass=negative_case_pass,
        passed=passed,
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
    )


def build_answer_eval_record(case: EvalCase, answer: Answer, model_config: dict) -> dict:
    evaluation = evaluate_answer(case, answer)
    return {
        "id": case.id,
        "question": case.question,
        "ground_truth": case.ground_truth,
        "generated_answer": answer.answer,
        "expected_answer_keywords": case.expected_answer_keywords,
        "expected_sources": case.expected_sources,
        "expected_pages": case.expected_pages,
        "expected_chunk_keywords": case.expected_chunk_keywords,
        "category": case.category,
        "difficulty": case.difficulty,
        "language": case.language,
        "is_negative": case.is_negative,
        "retrieved_contexts": [
            {
                "source": source.source,
                "page": source.page,
                "chunk_index": source.chunk_index,
                "content": source.content,
            }
            for source in answer.sources
        ],
        "sources": [source.source for source in answer.sources],
        "model_config": model_config,
        "metrics": {
            "keyword_coverage": evaluation.keyword_coverage,
            "citation_present": evaluation.citation_present,
            "answer_non_empty": evaluation.answer_non_empty,
            "unknown_answer": evaluation.unknown_answer,
            "negative_case_pass": evaluation.negative_case_pass,
            "passed": evaluation.passed,
            "matched_keywords": evaluation.matched_keywords,
            "missing_keywords": evaluation.missing_keywords,
        },
    }


def build_ragas_dataset_dict(records: list[dict]) -> dict:
    """Build the classic RAGAS dataset schema from answer-eval records."""
    return {
        "question": [record["question"] for record in records],
        "answer": [record["generated_answer"] for record in records],
        "contexts": [[context["content"] for context in record["retrieved_contexts"]] for record in records],
        "ground_truth": [record["ground_truth"] for record in records],
    }


def build_ragas_dataset(records: list[dict]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RagasEvaluationError(f"RAGAS dataset dependency is not installed: {exc.name}") from exc

    return Dataset.from_dict(build_ragas_dataset_dict(records))


def run_ragas_evaluation(
    records: list[dict],
    *,
    api_key: str,
    base_url: str = "",
    judge_model: str = "gpt-4o-mini",
    embedding_model: str = "bge-m3",
) -> dict:
    if not records:
        raise RagasEvaluationError("No answer evaluation records were provided")
    if not api_key:
        raise RagasEvaluationError("OPENAI_API_KEY is required for RAGAS answer evaluation judge")

    previous_api_key = os.environ.get("OPENAI_API_KEY")
    previous_base_url = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    try:
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        dataset = build_ragas_dataset(records)
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            model=judge_model,
            temperature=0,
        )
        if embedding_model in LOCAL_EMBEDDING_MODELS:
            embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
                model_name=LOCAL_EMBEDDING_MODELS[embedding_model],
                encode_kwargs={"normalize_embeddings": True},
            ))
        else:
            embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
                api_key=api_key,
                base_url=base_url or None,
                model=embedding_model,
            ))
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=True,
        )
    except ImportError as exc:
        raise RagasEvaluationError(f"RAGAS dependencies are required but not installed: {exc.name}") from exc
    except RagasEvaluationError:
        raise
    except Exception as exc:
        raise RagasEvaluationError(f"RAGAS evaluation failed: {type(exc).__name__}: {exc}") from exc
    finally:
        _restore_env("OPENAI_API_KEY", previous_api_key)
        _restore_env("OPENAI_BASE_URL", previous_base_url)

    return {
        "enabled": True,
        "required": True,
        "metrics": _serialize_ragas_result(result),
        "metric_names": RAGAS_METRIC_NAMES,
        "target_thresholds": RAGAS_TARGET_THRESHOLDS,
        "dataset_schema": list(build_ragas_dataset_dict(records).keys()),
        "judge_config": {
            "base_url": base_url,
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "judge_source": "business_qa_openai_config",
        },
    }


def summarize_answer_eval(records: list[dict]) -> dict:
    if not records:
        return {
            "cases": 0,
            "pass_rate": 0.0,
            "average_keyword_coverage": 0.0,
            "citation_rate": 0.0,
            "unknown_answer_rate": 0.0,
            "negative_case_pass_rate": 0.0,
        }

    metrics = [record["metrics"] for record in records]
    negative_metrics = [metric for metric in metrics if metric["negative_case_pass"] is not None]
    return {
        "cases": len(records),
        "pass_rate": mean(1.0 if metric["passed"] else 0.0 for metric in metrics),
        "average_keyword_coverage": mean(metric["keyword_coverage"] for metric in metrics),
        "citation_rate": mean(1.0 if metric["citation_present"] else 0.0 for metric in metrics),
        "unknown_answer_rate": mean(1.0 if metric["unknown_answer"] else 0.0 for metric in metrics),
        "negative_case_pass_rate": (
            mean(1.0 if metric["negative_case_pass"] else 0.0 for metric in negative_metrics)
            if negative_metrics
            else 0.0
        ),
    }


def is_unknown_answer(answer: str) -> bool:
    lowered = answer.lower()
    if not lowered.strip():
        return True
    return any(marker in lowered for marker in UNKNOWN_MARKERS)


def _restore_env(name: str, previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous_value


def _serialize_ragas_result(result) -> dict:
    if hasattr(result, "_repr_dict"):
        return _jsonable(result._repr_dict)
    if hasattr(result, "scores") and result.scores:
        return _jsonable({key: mean(row[key] for row in result.scores) for key in result.scores[0]})

    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return _jsonable(frame.mean(numeric_only=True).to_dict())

    try:
        return _jsonable(dict(result))
    except (KeyError, TypeError, ValueError):
        pass

    return {"raw": str(result)}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
