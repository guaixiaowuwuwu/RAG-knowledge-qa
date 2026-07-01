import json
from pathlib import Path

import pytest

from app.ingestion.chunker import Chunk
from app.evaluation.dataset import load_eval_cases
from app.rag.confidence import REFUSAL_ANSWER, RetrievalConfidenceConfig
from app.rag.service import RagService


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_query = None
        self.last_top_k = None

    def similarity_search(self, query: str, top_k: int):
        self.last_query = query
        self.last_top_k = top_k
        return self.chunks[:top_k]


class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "这是基于知识库的回答。"


class FakeTrace:
    def __init__(self, reranker_scores=None):
        self.reranker_scores = reranker_scores or []

    def to_dict(self):
        return {
            "query_variants": ["系统是什么？"],
            "rrf_scores": [{"source": "trace.md", "score": 0.1}],
            "reranker_scores": self.reranker_scores,
        }


class FakeTraceResult:
    def __init__(self, chunks, trace=None):
        self.chunks = chunks
        self.trace = trace or FakeTrace()


class FakeTraceRetriever(FakeRetriever):
    def __init__(self, chunks, trace=None):
        super().__init__(chunks)
        self.trace = trace or FakeTrace()

    def similarity_search_with_trace(self, query: str, top_k: int, options=None):
        self.last_query = query
        self.last_top_k = top_k
        self.last_options = options
        return FakeTraceResult(self.chunks[:top_k], trace=self.trace)


NEGATIVE_EVAL_CASES = [
    case
    for case in load_eval_cases(Path("data/eval/sample_eval.jsonl"))
    if case.is_negative
]


def test_answer_returns_no_context_message_when_retrieval_is_empty():
    retriever = FakeRetriever([])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    response = service.answer("系统是什么？", top_k=3)

    assert response.answer == REFUSAL_ANSWER
    assert response.sources == []
    assert llm.last_prompt is None


def test_answer_calls_llm_with_context_and_returns_sources():
    chunk = Chunk(
        content="RAG 系统包含文档解析和向量检索。",
        source="data/documents/example.md",
        metadata={"source": "data/documents/example.md", "chunk_index": 0, "file_type": ".md"},
    )
    retriever = FakeRetriever([chunk])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    response = service.answer("RAG 系统包含什么？", top_k=5)

    assert retriever.last_query == "RAG 系统包含什么？"
    assert retriever.last_top_k == 5
    assert "RAG 系统包含文档解析和向量检索。" in llm.last_prompt
    assert "RAG 系统包含什么？" in llm.last_prompt
    assert response.answer == "这是基于知识库的回答。"
    assert response.sources[0].source == "data/documents/example.md"
    assert response.sources[0].chunk_index == 0
    assert response.sources[0].content == "RAG 系统包含文档解析和向量检索。"


def test_answer_prompt_tells_llm_to_avoid_context_preface():
    chunk = Chunk(
        content="比亚迪制定了信息披露事务管理制度。",
        source="data/documents/byd_chinese/policy.pdf",
        metadata={"source": "data/documents/byd_chinese/policy.pdf", "chunk_index": 0, "file_type": ".pdf"},
    )
    retriever = FakeRetriever([chunk])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    service.answer("制度主要规定了什么？", top_k=3)

    assert "不要使用" in llm.last_prompt
    assert "根据上下文信息" in llm.last_prompt
    assert "直接给出结论或要点" in llm.last_prompt


def test_answer_prompt_tells_llm_to_preserve_original_numbers_and_units():
    chunk = Chunk(
        content="本集团营业额约人民币 777,102 百万元，同比上升 29.02%。",
        source="data/documents/byd_chinese/report.pdf",
        metadata={"source": "data/documents/byd_chinese/report.pdf", "chunk_index": 0, "file_type": ".pdf"},
    )
    retriever = FakeRetriever([chunk])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    service.answer("营业额和同比增幅是多少？", top_k=3)

    assert "优先保留上下文中的原始数值、单位和格式" in llm.last_prompt
    assert "不要只给换算后的表达" in llm.last_prompt


def test_answer_prompt_rejects_unsupported_realtime_or_private_inference():
    chunk = Chunk(
        content="比亚迪年报披露了年度经营情况。",
        source="data/documents/byd_chinese/report.pdf",
        metadata={"source": "data/documents/byd_chinese/report.pdf", "chunk_index": 0, "file_type": ".pdf"},
    )
    retriever = FakeRetriever([chunk])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    service.answer("比亚迪年度经营情况是什么？", top_k=3)

    assert "不能用来推断实时股价、当日行情、实时产量、私密合同或未披露信息" in llm.last_prompt


@pytest.mark.parametrize("case", NEGATIVE_EVAL_CASES, ids=lambda case: case.id)
def test_answer_refuses_negative_eval_cases_without_sources(case):
    misleading_chunk = Chunk(
        content="比亚迪、微软、英伟达和苹果公开报告披露了部分年度经营数据。",
        source="data/documents/byd_chinese/BYD_2025_annual_report_cn.pdf",
        metadata={"source": "data/documents/byd_chinese/BYD_2025_annual_report_cn.pdf", "chunk_index": 0},
    )
    retriever = FakeRetriever([misleading_chunk])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    response = service.answer(case.question, top_k=3, debug=True)

    assert response.answer == REFUSAL_ANSWER
    assert response.sources == []
    assert llm.last_prompt is None
    assert response.debug["confidence"]["should_refuse"] is True
    assert response.debug["confidence"]["refusal_reason"] in {
        "time_sensitive",
        "private_or_unavailable",
        "low_retrieval_confidence",
        "empty_retrieval",
    }


def test_answer_can_return_debug_trace_when_requested():
    chunk = Chunk(
        content="RAG 系统包含检索和生成。",
        source="data/documents/example.md",
        metadata={"source": "data/documents/example.md", "chunk_index": 0},
    )
    retriever = FakeTraceRetriever([chunk])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    response = service.answer("系统是什么？", top_k=3, debug=True, retrieval_options=object())

    assert response.debug["query_variants"] == ["系统是什么？"]
    assert response.debug["rrf_scores"] == [{"source": "trace.md", "score": 0.1}]
    assert response.debug["confidence"]["should_refuse"] is False
    assert response.debug["timings_ms"]["retrieval"] >= 0
    assert retriever.last_options is not None


def test_answer_refuses_when_reranker_score_is_below_threshold():
    chunk = Chunk(
        content="不相关的检索结果。",
        source="data/documents/example.md",
        metadata={"source": "data/documents/example.md", "chunk_index": 0},
    )
    trace = FakeTrace(reranker_scores=[{"source": "data/documents/example.md", "score": -2.0}])
    retriever = FakeTraceRetriever([chunk], trace=trace)
    llm = FakeLLM()
    service = RagService(
        retriever=retriever,
        llm=llm,
        confidence_config=RetrievalConfidenceConfig(min_reranker_score=0.0),
    )

    response = service.answer("系统是什么？", top_k=3, debug=True)

    assert response.answer == REFUSAL_ANSWER
    assert response.sources == []
    assert llm.last_prompt is None
    assert response.debug["confidence"]["refusal_reason"] == "low_retrieval_confidence"
    assert response.debug["confidence"]["best_reranker_score"] == -2.0


def test_answer_stream_emits_debug_event_when_requested():
    chunk = Chunk(
        content="RAG 系统包含检索和生成。",
        source="data/documents/example.md",
        metadata={"source": "data/documents/example.md", "chunk_index": 0},
    )
    service = RagService(retriever=FakeTraceRetriever([chunk]), llm=FakeLLM())

    events = list(service.answer_stream("系统是什么？", top_k=3, debug=True))

    assert events[-1]["event"] == "debug"
    assert "query_variants" in events[-1]["data"]
    assert "timings_ms" in events[-1]["data"]


def test_answer_stream_refusal_emits_empty_sources_and_debug_reason():
    chunk = Chunk(
        content="比亚迪年报披露年度经营数据。",
        source="data/documents/byd_chinese/BYD_2025_annual_report_cn.pdf",
        metadata={"source": "data/documents/byd_chinese/BYD_2025_annual_report_cn.pdf", "chunk_index": 0},
    )
    service = RagService(retriever=FakeRetriever([chunk]), llm=FakeLLM())

    events = list(service.answer_stream("比亚迪今天的股票收盘价是多少？", top_k=3, debug=True))

    assert events[0] == {"event": "token", "data": REFUSAL_ANSWER}
    assert events[1] == {"event": "sources", "data": "[]"}
    debug_payload = json.loads(events[2]["data"])
    assert debug_payload["confidence"]["refusal_reason"] == "time_sensitive"
