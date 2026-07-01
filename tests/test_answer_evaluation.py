import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_answers
from app.evaluation.answer_eval import (
    RAGAS_TARGET_THRESHOLDS,
    RagasEvaluationError,
    _serialize_ragas_result,
    build_answer_eval_record,
    build_ragas_dataset_dict,
    evaluate_answer,
    run_ragas_evaluation,
    summarize_answer_eval,
)
from app.evaluation.dataset import EvalCase
from app.rag.service import Answer, Source


def test_answer_eval_scores_keyword_coverage_and_citation():
    case = EvalCase(
        id="case-1",
        question="RAG 是什么？",
        ground_truth="RAG 包含检索和生成。",
        expected_sources=["rag.md"],
        expected_answer_keywords=["检索", "生成"],
    )
    answer = Answer(
        answer="RAG 系统先检索知识库，再生成回答。",
        sources=[Source(source="rag.md", page=None, chunk_index=0, content="检索和生成")],
    )

    result = evaluate_answer(case, answer)

    assert result.keyword_coverage == 1.0
    assert result.citation_present is True
    assert result.passed is True


def test_answer_eval_handles_negative_case():
    case = EvalCase(
        id="negative-1",
        question="知识库有火星基地预算吗？",
        ground_truth="知识库没有该信息。",
        expected_sources=[],
        expected_answer_keywords=["没有"],
        is_negative=True,
    )
    answer = Answer(answer="知识库中没有找到相关内容，无法基于现有资料回答。", sources=[])

    result = evaluate_answer(case, answer)

    assert result.unknown_answer is True
    assert result.negative_case_pass is True
    assert result.passed is True


def test_answer_eval_record_and_summary():
    case = EvalCase(
        id="case-1",
        question="RAG 是什么？",
        ground_truth="RAG 包含检索。",
        expected_sources=["rag.md"],
        expected_answer_keywords=["检索"],
    )
    answer = Answer(
        answer="RAG 包含检索。",
        sources=[Source(source="rag.md", page=1, chunk_index=0, content="RAG 包含检索。")],
    )

    record = build_answer_eval_record(case, answer, {"chat_model": "fake"})
    summary = summarize_answer_eval([record])

    assert record["retrieved_contexts"][0]["source"] == "rag.md"
    assert summary["cases"] == 1
    assert summary["pass_rate"] == 1.0


def test_build_ragas_dataset_dict_uses_classic_ragas_schema():
    case = EvalCase(
        id="case-1",
        question="RAG 是什么？",
        ground_truth="RAG 包含检索和生成。",
        expected_sources=["rag.md"],
        expected_answer_keywords=["检索", "生成"],
    )
    answer = Answer(
        answer="RAG 包含检索和生成。",
        sources=[Source(source="rag.md", page=1, chunk_index=0, content="RAG 先检索，再生成。")],
    )
    record = build_answer_eval_record(case, answer, {"chat_model": "fake"})

    dataset = build_ragas_dataset_dict([record])

    assert list(dataset) == ["question", "answer", "contexts", "ground_truth"]
    assert dataset["question"] == ["RAG 是什么？"]
    assert dataset["answer"] == ["RAG 包含检索和生成。"]
    assert dataset["contexts"] == [["RAG 先检索，再生成。"]]
    assert dataset["ground_truth"] == ["RAG 包含检索和生成。"]


def test_ragas_evaluation_requires_api_key():
    with pytest.raises(RagasEvaluationError, match="OPENAI_API_KEY is required"):
        run_ragas_evaluation(
            [{"question": "q", "generated_answer": "a", "retrieved_contexts": [], "ground_truth": "g"}],
            api_key="",
        )


def test_ragas_evaluation_requires_records():
    with pytest.raises(RagasEvaluationError, match="No answer evaluation records"):
        run_ragas_evaluation([], api_key="key")


def test_serialize_ragas_result_handles_evaluation_result_like_object():
    class FakeRagasResult:
        _repr_dict = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
        }

        def __getitem__(self, key):
            raise KeyError(key)

    assert _serialize_ragas_result(FakeRagasResult()) == {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
    }


def test_ragas_evaluation_restores_openai_env_after_failure(monkeypatch):
    monkeypatch.setitem(os.environ, "OPENAI_API_KEY", "previous-key")
    monkeypatch.setitem(os.environ, "OPENAI_BASE_URL", "https://previous.example/v1")
    monkeypatch.setattr("app.evaluation.answer_eval.build_ragas_dataset", lambda records: object())

    fake_ragas = types.ModuleType("ragas")
    fake_ragas_embeddings = types.ModuleType("ragas.embeddings")
    fake_metrics = types.ModuleType("ragas.metrics")
    fake_langchain_community_embeddings = types.ModuleType("langchain_community.embeddings")
    fake_langchain_openai = types.ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            pass

    class FakeLangchainEmbeddingsWrapper:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            pass

    def fail_evaluate(dataset, metrics, llm, embeddings, raise_exceptions):
        assert os.environ["OPENAI_API_KEY"] == "new-key"
        assert os.environ["OPENAI_BASE_URL"] == "https://new.example/v1"
        assert raise_exceptions is True
        raise RuntimeError("boom")

    fake_ragas.evaluate = fail_evaluate
    fake_ragas_embeddings.LangchainEmbeddingsWrapper = FakeLangchainEmbeddingsWrapper
    fake_metrics.faithfulness = object()
    fake_metrics.answer_relevancy = object()
    fake_metrics.context_precision = object()
    fake_metrics.context_recall = object()
    fake_langchain_community_embeddings.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
    fake_langchain_openai.ChatOpenAI = FakeChatOpenAI
    fake_langchain_openai.OpenAIEmbeddings = FakeOpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", fake_ragas_embeddings)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics)
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", fake_langchain_community_embeddings)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_langchain_openai)

    with pytest.raises(RagasEvaluationError, match="RuntimeError: boom"):
        run_ragas_evaluation(
            [{"question": "q", "generated_answer": "a", "retrieved_contexts": [], "ground_truth": "g"}],
            api_key="new-key",
            base_url="https://new.example/v1",
        )

    assert os.environ["OPENAI_API_KEY"] == "previous-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://previous.example/v1"
    assert RAGAS_TARGET_THRESHOLDS["faithfulness"] == 0.85


def test_ragas_evaluation_invokes_required_metrics(monkeypatch):
    monkeypatch.setattr("app.evaluation.answer_eval.build_ragas_dataset", lambda records: {"rows": records})

    fake_ragas = types.ModuleType("ragas")
    fake_ragas_embeddings = types.ModuleType("ragas.embeddings")
    fake_metrics = types.ModuleType("ragas.metrics")
    fake_langchain_community_embeddings = types.ModuleType("langchain_community.embeddings")
    fake_langchain_openai = types.ModuleType("langchain_openai")
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for metric_name in metric_names:
        setattr(fake_metrics, metric_name, metric_name)

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            captured["embedding_kwargs"] = kwargs

    class FakeLangchainEmbeddingsWrapper:
        def __init__(self, embeddings):
            captured["wrapped_embeddings"] = embeddings

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            captured["embedding_kwargs"] = kwargs

    def fake_evaluate(dataset, metrics, llm, embeddings, raise_exceptions):
        captured["dataset"] = dataset
        captured["metrics"] = metrics
        captured["llm"] = llm
        captured["embeddings"] = embeddings
        captured["raise_exceptions"] = raise_exceptions
        return {
            "faithfulness": 0.9,
            "answer_relevancy": 0.88,
            "context_precision": 0.8,
            "context_recall": 0.82,
        }

    fake_ragas.evaluate = fake_evaluate
    fake_ragas_embeddings.LangchainEmbeddingsWrapper = FakeLangchainEmbeddingsWrapper
    fake_langchain_community_embeddings.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
    fake_langchain_openai.ChatOpenAI = FakeChatOpenAI
    fake_langchain_openai.OpenAIEmbeddings = FakeOpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", fake_ragas_embeddings)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics)
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", fake_langchain_community_embeddings)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_langchain_openai)

    result = run_ragas_evaluation(
        [{"question": "q", "generated_answer": "a", "retrieved_contexts": [], "ground_truth": "g"}],
        api_key="key",
        base_url="https://judge.example/v1",
        judge_model="judge-model",
    )

    assert result["enabled"] is True
    assert result["required"] is True
    assert result["metrics"]["faithfulness"] == 0.9
    assert result["target_thresholds"] == RAGAS_TARGET_THRESHOLDS
    assert result["judge_config"]["judge_model"] == "judge-model"
    assert result["judge_config"]["embedding_model"] == "bge-m3"
    assert result["judge_config"]["judge_source"] == "business_qa_openai_config"
    assert captured["llm_kwargs"]["model"] == "judge-model"
    assert captured["embedding_kwargs"]["model_name"] == "BAAI/bge-m3"
    assert captured["embedding_kwargs"]["encode_kwargs"] == {"normalize_embeddings": True}
    assert captured["metrics"] == metric_names
    assert captured["raise_exceptions"] is True


def test_evaluate_answers_main_uses_business_judge_and_bge_m3(monkeypatch, tmp_path: Path):
    output = tmp_path / "answer-eval.json"
    captured = {}

    settings = SimpleNamespace(
        eval_dataset_path=tmp_path / "eval.jsonl",
        retrieval_top_k=4,
        openai_api_key="answer-key",
        openai_base_url="https://answer.example/v1",
        chat_model="answer-model",
    )
    case = EvalCase(
        id="case-1",
        question="RAG 是什么？",
        ground_truth="RAG 包含检索。",
        expected_sources=["rag.md"],
        expected_answer_keywords=["检索"],
    )

    class FakeService:
        def answer(self, question, top_k):
            return Answer(
                answer="RAG 包含检索。",
                sources=[Source(source="rag.md", page=None, chunk_index=0, content="RAG 包含检索。")],
            )

    def fake_ragas(records, **kwargs):
        captured["records"] = records
        captured["kwargs"] = kwargs
        return {"enabled": True, "required": True, "metrics": {"faithfulness": 1.0}}

    monkeypatch.setattr(evaluate_answers, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluate_answers, "load_eval_cases", lambda path: [case])
    monkeypatch.setattr(evaluate_answers, "config_snapshot", lambda settings: {})
    monkeypatch.setattr(evaluate_answers, "build_rag_service", lambda: FakeService())
    monkeypatch.setattr(evaluate_answers, "run_ragas_evaluation", fake_ragas)

    evaluate_answers.main(["--output", str(output)])

    assert captured["kwargs"] == {
        "api_key": "answer-key",
        "base_url": "https://answer.example/v1",
        "judge_model": "answer-model",
        "embedding_model": "bge-m3",
    }
    assert captured["records"][0]["generated_answer"] == "RAG 包含检索。"
    assert output.exists()


def test_evaluate_answers_main_ignores_ragas_specific_attributes(monkeypatch, tmp_path: Path):
    output = tmp_path / "answer-eval.json"
    captured = {}

    settings = SimpleNamespace(
        eval_dataset_path=tmp_path / "eval.jsonl",
        retrieval_top_k=4,
        openai_api_key="business-key",
        openai_base_url="https://business.example/v1",
        chat_model="business-chat",
        ragas_api_key="ignored-key",
        ragas_base_url="https://ignored.example/v1",
        ragas_judge_model="ignored-model",
        ragas_embedding_model="ignored-embedding",
    )
    case = EvalCase(
        id="case-1",
        question="RAG 是什么？",
        ground_truth="RAG 包含检索。",
        expected_sources=["rag.md"],
        expected_answer_keywords=["检索"],
    )

    class FakeService:
        def answer(self, question, top_k):
            return Answer(
                answer="RAG 包含检索。",
                sources=[Source(source="rag.md", page=None, chunk_index=0, content="RAG 包含检索。")],
            )

    def fake_ragas(records, **kwargs):
        captured["kwargs"] = kwargs
        return {"enabled": True, "required": True, "metrics": {"faithfulness": 1.0}}

    monkeypatch.setattr(evaluate_answers, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluate_answers, "load_eval_cases", lambda path: [case])
    monkeypatch.setattr(evaluate_answers, "config_snapshot", lambda settings: {})
    monkeypatch.setattr(evaluate_answers, "build_rag_service", lambda: FakeService())
    monkeypatch.setattr(evaluate_answers, "run_ragas_evaluation", fake_ragas)

    evaluate_answers.main(["--output", str(output)])

    assert captured["kwargs"] == {
        "api_key": "business-key",
        "base_url": "https://business.example/v1",
        "judge_model": "business-chat",
        "embedding_model": "bge-m3",
    }


def test_select_answer_eval_cases_supports_mixed_category_and_language_sampling():
    cases = [
        EvalCase(id="zh-fact", question="q", ground_truth="g", expected_sources=["a"], category="exact_fact", language="zh"),
        EvalCase(id="en-fact", question="q", ground_truth="g", expected_sources=["a"], category="exact_fact", language="en"),
        EvalCase(id="en-risk", question="q", ground_truth="g", expected_sources=["a"], category="risk", language="en"),
        EvalCase(id="en-negative", question="q", ground_truth="g", expected_sources=[], category="negative", language="en", is_negative=True),
        EvalCase(id="en-policy", question="q", ground_truth="g", expected_sources=["a"], category="policy", language="en"),
    ]

    selected = evaluate_answers.select_answer_eval_cases(
        cases,
        limit=3,
        sample="mixed",
        include_categories=["exact_fact", "risk", "negative"],
        language="en",
        random_seed=None,
    )

    assert [case.id for case in selected] == ["en-fact", "en-risk", "en-negative"]


def test_evaluate_answers_main_can_skip_ragas_for_local_diagnosis(monkeypatch, tmp_path: Path):
    output = tmp_path / "answer-eval.json"
    settings = SimpleNamespace(
        eval_dataset_path=tmp_path / "eval.jsonl",
        retrieval_top_k=4,
        openai_api_key="",
        openai_base_url="https://answer.example/v1",
        chat_model="answer-model",
    )
    case = EvalCase(
        id="case-1",
        question="RAG 是什么？",
        ground_truth="RAG 包含检索。",
        expected_sources=["rag.md"],
        expected_answer_keywords=["检索"],
        category="exact_fact",
        language="zh",
    )

    class FakeService:
        def answer(self, question, top_k):
            return Answer(
                answer="RAG 包含检索。",
                sources=[Source(source="rag.md", page=None, chunk_index=0, content="RAG 包含检索。")],
            )

    def fail_ragas(*args, **kwargs):
        raise AssertionError("RAGAS should be skipped")

    monkeypatch.setattr(evaluate_answers, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluate_answers, "load_eval_cases", lambda path: [case])
    monkeypatch.setattr(evaluate_answers, "config_snapshot", lambda settings: {})
    monkeypatch.setattr(evaluate_answers, "build_rag_service", lambda: FakeService())
    monkeypatch.setattr(evaluate_answers, "run_ragas_evaluation", fail_ragas)

    evaluate_answers.main(["--output", str(output), "--no-ragas"])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["cases"] == 1
    assert report["ragas"]["enabled"] is False
    assert report["ragas"]["required"] is False


def test_evaluate_answers_report_config_includes_effective_index_path(monkeypatch, tmp_path: Path):
    output = tmp_path / "answer-eval.json"
    settings = SimpleNamespace(
        eval_dataset_path=tmp_path / "eval.jsonl",
        retrieval_top_k=4,
        openai_api_key="",
        openai_base_url="https://answer.example/v1",
        chat_model="answer-model",
        embedding_model="bge-m3",
        chroma_collection="test",
        chroma_dir=tmp_path / "legacy" / "chroma",
        bm25_corpus_path=tmp_path / "legacy" / "bm25.jsonl",
        parent_corpus_path=tmp_path / "legacy" / "parents.jsonl",
        index_root_dir=tmp_path / "indexes",
        active_index_version_path=tmp_path / "indexes" / "active_version.txt",
        document_index_version="configured",
        versioned_indexing_enabled=True,
    )
    case = EvalCase(
        id="case-1",
        question="RAG 是什么？",
        ground_truth="RAG 包含检索。",
        expected_sources=["rag.md"],
        expected_answer_keywords=["检索"],
    )

    class FakeService:
        def answer(self, question, top_k):
            return Answer(
                answer="RAG 包含检索。",
                sources=[Source(source="rag.md", page=None, chunk_index=0, content="RAG 包含检索。")],
            )

    monkeypatch.setattr(evaluate_answers, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluate_answers, "load_eval_cases", lambda path: [case])
    monkeypatch.setattr(evaluate_answers, "build_rag_service", lambda: FakeService())

    evaluate_answers.main(["--output", str(output), "--no-ragas"])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["effective_bm25_corpus_path"] == str(
        tmp_path / "indexes" / "configured" / "bm25_corpus.jsonl"
    )
    assert report["config"]["effective_parent_corpus_path"] == str(
        tmp_path / "indexes" / "configured" / "parent_corpus.jsonl"
    )


def test_evaluate_answers_main_saves_failure_report_for_generation_error(monkeypatch, tmp_path: Path):
    output = tmp_path / "answer-eval-failed.json"
    settings = SimpleNamespace(
        eval_dataset_path=tmp_path / "eval.jsonl",
        retrieval_top_k=4,
        openai_api_key="",
        openai_base_url="https://answer.example/v1",
        chat_model="answer-model",
    )
    case = EvalCase(
        id="case-1",
        question="RAG 是什么？",
        ground_truth="RAG 包含检索。",
        expected_sources=["rag.md"],
        expected_answer_keywords=["检索"],
    )

    monkeypatch.setattr(evaluate_answers, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluate_answers, "load_eval_cases", lambda path: [case])
    monkeypatch.setattr(evaluate_answers, "config_snapshot", lambda settings: {})
    monkeypatch.setattr(evaluate_answers, "build_rag_service", lambda: (_ for _ in ()).throw(ValueError("missing key")))

    with pytest.raises(SystemExit, match="Answer generation failed"):
        evaluate_answers.main(["--output", str(output)])

    contents = output.read_text(encoding="utf-8")
    assert "Answer generation failed: ValueError: missing key" in contents
    assert '"records": []' in contents
