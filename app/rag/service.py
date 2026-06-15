import json
import time
from dataclasses import dataclass
from typing import Protocol

from app.ingestion.chunker import Chunk
from app.rag.prompts import build_rag_prompt
from app.rag.hybrid_retriever import RetrievalDebugOptions, RetrievalTrace


@dataclass(frozen=True)
class Source:
    source: str
    page: int | None
    chunk_index: int | None
    content: str
    matched_child_chunk_index: int | None = None
    content_type: str | None = None
    table_index: int | None = None


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source]
    debug: dict | None = None


class Retriever(Protocol):
    def similarity_search(self, query: str, top_k: int) -> list[Chunk]:
        ...


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class RagService:
    def __init__(self, retriever: Retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    def answer(
        self,
        question: str,
        top_k: int,
        *,
        debug: bool = False,
        retrieval_options: RetrievalDebugOptions | None = None,
    ) -> Answer:
        retrieval_started_at = time.perf_counter()
        chunks, trace = self._retrieve(question, top_k=top_k, debug=debug, retrieval_options=retrieval_options)
        retrieval_ms = elapsed_ms(retrieval_started_at)
        if not chunks:
            debug_payload = trace.to_dict() if debug and trace is not None else None
            if debug_payload is not None:
                debug_payload["timings_ms"] = {"retrieval": retrieval_ms, "llm": 0.0, "total": retrieval_ms}
            return Answer(
                answer="知识库中没有找到相关内容，无法基于现有资料回答。",
                sources=[],
                debug=debug_payload,
            )

        prompt = build_rag_prompt(question, chunks)
        llm_started_at = time.perf_counter()
        answer = self.llm.complete(prompt)
        llm_ms = elapsed_ms(llm_started_at)
        debug_payload = trace.to_dict() if debug and trace is not None else None
        if debug_payload is not None:
            debug_payload["timings_ms"] = {
                "retrieval": retrieval_ms,
                "llm": llm_ms,
                "total": round(retrieval_ms + llm_ms, 3),
            }
        sources = [
            Source(
                source=chunk.source,
                page=chunk.metadata.get("page"),
                chunk_index=chunk.metadata.get("chunk_index"),
                content=chunk.content,
                matched_child_chunk_index=chunk.metadata.get("matched_child_chunk_index"),
                content_type=chunk.metadata.get("content_type"),
                table_index=chunk.metadata.get("table_index"),
            )
            for chunk in chunks
        ]
        return Answer(answer=answer, sources=sources, debug=debug_payload)

    def answer_stream(
        self,
        question: str,
        top_k: int,
        *,
        debug: bool = False,
        retrieval_options: RetrievalDebugOptions | None = None,
    ):
        retrieval_started_at = time.perf_counter()
        chunks, trace = self._retrieve(question, top_k=top_k, debug=debug, retrieval_options=retrieval_options)
        retrieval_ms = elapsed_ms(retrieval_started_at)
        if not chunks:
            yield {"event": "token", "data": "知识库中没有找到相关内容，无法基于现有资料回答。"}
            yield {"event": "sources", "data": "[]"}
            if debug and trace is not None:
                debug_payload = trace.to_dict()
                debug_payload["timings_ms"] = {"retrieval": retrieval_ms, "llm": 0.0, "total": retrieval_ms}
                yield {"event": "debug", "data": json.dumps(debug_payload, ensure_ascii=False)}
            return

        prompt = build_rag_prompt(question, chunks)
        llm_started_at = time.perf_counter()
        stream = self.llm.stream(prompt) if hasattr(self.llm, "stream") else [self.llm.complete(prompt)]
        for token in stream:
            yield {"event": "token", "data": token}
        llm_ms = elapsed_ms(llm_started_at)

        sources = [
            {
                "source": chunk.source,
                "page": chunk.metadata.get("page"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "matched_child_chunk_index": chunk.metadata.get("matched_child_chunk_index"),
                "content_type": chunk.metadata.get("content_type"),
                "table_index": chunk.metadata.get("table_index"),
                "content": chunk.content,
            }
            for chunk in chunks
        ]
        yield {"event": "sources", "data": json.dumps(sources, ensure_ascii=False)}
        if debug and trace is not None:
            debug_payload = trace.to_dict()
            debug_payload["timings_ms"] = {
                "retrieval": retrieval_ms,
                "llm": llm_ms,
                "total": round(retrieval_ms + llm_ms, 3),
            }
            yield {"event": "debug", "data": json.dumps(debug_payload, ensure_ascii=False)}

    def _retrieve(
        self,
        question: str,
        *,
        top_k: int,
        debug: bool,
        retrieval_options: RetrievalDebugOptions | None,
    ) -> tuple[list[Chunk], RetrievalTrace | None]:
        if debug and hasattr(self.retriever, "similarity_search_with_trace"):
            result = self.retriever.similarity_search_with_trace(
                question,
                top_k=top_k,
                options=retrieval_options,
            )
            return result.chunks, result.trace

        return self.retriever.similarity_search(question, top_k=top_k), None


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)
