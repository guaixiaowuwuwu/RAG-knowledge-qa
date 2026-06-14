from app.ingestion.chunker import Chunk
from app.rag.service import RagService


class FakeRetriever:
    def similarity_search(self, query: str, top_k: int):
        return [
            Chunk(
                content="RAG 系统支持流式输出。",
                source="stream.md",
                metadata={"source": "stream.md", "chunk_index": 0},
            )
        ]


class FakeStreamingLLM:
    def stream(self, prompt: str):
        yield "第一段"
        yield "第二段"

    def complete(self, prompt: str) -> str:
        return "第一段第二段"


def test_answer_stream_yields_text_chunks_then_sources_event():
    service = RagService(retriever=FakeRetriever(), llm=FakeStreamingLLM())

    events = list(service.answer_stream("怎么流式输出？", top_k=4))

    assert events[0] == {"event": "token", "data": "第一段"}
    assert events[1] == {"event": "token", "data": "第二段"}
    assert events[2]["event"] == "sources"
    assert "stream.md" in events[2]["data"]
