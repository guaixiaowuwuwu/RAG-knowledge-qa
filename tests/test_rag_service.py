from app.ingestion.chunker import Chunk
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
    def to_dict(self):
        return {"query_variants": ["系统是什么？"], "rrf_scores": [{"source": "trace.md", "score": 0.1}]}


class FakeTraceResult:
    def __init__(self, chunks):
        self.chunks = chunks
        self.trace = FakeTrace()


class FakeTraceRetriever(FakeRetriever):
    def similarity_search_with_trace(self, query: str, top_k: int, options=None):
        self.last_query = query
        self.last_top_k = top_k
        self.last_options = options
        return FakeTraceResult(self.chunks[:top_k])


def test_answer_returns_no_context_message_when_retrieval_is_empty():
    retriever = FakeRetriever([])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    response = service.answer("系统是什么？", top_k=3)

    assert response.answer == "知识库中没有找到相关内容，无法基于现有资料回答。"
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
    assert response.debug["timings_ms"]["retrieval"] >= 0
    assert retriever.last_options is not None


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
