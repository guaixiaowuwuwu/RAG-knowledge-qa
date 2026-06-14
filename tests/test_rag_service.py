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
