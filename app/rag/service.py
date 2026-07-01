import json
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from app.audit.repository import AuditRepository
from app.ingestion.chunker import Chunk
from app.rag.confidence import (
    REFUSAL_ANSWER,
    RetrievalConfidenceConfig,
    RetrievalConfidenceDecision,
    decide_retrieval_confidence,
)
from app.rag.prompts import build_rag_prompt
from app.rag.hybrid_retriever import RetrievalDebugOptions, RetrievalTrace
from app.observability.metrics import metrics
from app.security.acl import RetrievalAccessFilter
from app.security.context import RequestContext


logger = logging.getLogger(__name__)
LLM_UNAVAILABLE_ANSWER = "回答服务暂时不可用，请稍后重试。"


@dataclass(frozen=True)
class Source:
    source: str
    page: int | None
    chunk_index: int | None
    content: str
    matched_child_chunk_index: int | None = None
    content_type: str | None = None
    table_index: int | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source]
    debug: dict | None = None
    session_id: str | None = None
    refusal_reason: str | None = None


class Retriever(Protocol):
    def similarity_search(self, query: str, top_k: int) -> list[Chunk]:
        ...


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class RagService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        confidence_config: RetrievalConfidenceConfig | None = None,
        audit_repository: AuditRepository | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.confidence_config = confidence_config or RetrievalConfidenceConfig()
        self.audit_repository = audit_repository

    def answer(
        self,
        question: str,
        top_k: int,
        *,
        debug: bool = False,
        retrieval_options: RetrievalDebugOptions | None = None,
        context: RequestContext | None = None,
        session_id: str | None = None,
    ) -> Answer:
        started_at = time.perf_counter()
        retrieval_started_at = started_at
        chunks, trace = self._retrieve(
            question,
            top_k=top_k,
            debug=debug,
            retrieval_options=retrieval_options,
            context=context,
        )
        retrieval_ms = elapsed_ms(retrieval_started_at)
        metrics.observe("rag_retrieval_latency_ms", retrieval_ms)
        decision = decide_retrieval_confidence(
            question,
            chunks,
            trace=trace,
            config=self.confidence_config,
        )
        if decision.should_refuse:
            total_ms = elapsed_ms(started_at)
            debug_payload = self._debug_payload(
                debug=debug,
                trace=trace,
                decision=decision,
                timings_ms={"retrieval": retrieval_ms, "llm": 0.0, "total": total_ms},
            )
            recorded_session_id = self._record_audit(
                question=question,
                answer=REFUSAL_ANSWER,
                sources=[],
                context=context,
                refusal_reason=decision.refusal_reason,
                latency_ms=total_ms,
                session_id=session_id,
            )
            return Answer(
                answer=REFUSAL_ANSWER,
                sources=[],
                debug=debug_payload,
                session_id=recorded_session_id,
                refusal_reason=decision.refusal_reason,
            )

        prompt = build_rag_prompt(question, chunks)
        llm_started_at = time.perf_counter()
        try:
            answer = self.llm.complete(prompt)
            llm_ms = elapsed_ms(llm_started_at)
        except Exception as exc:
            llm_ms = elapsed_ms(llm_started_at)
            total_ms = elapsed_ms(started_at)
            metrics.increment("rag_errors_total", stage="llm")
            metrics.observe("rag_llm_latency_ms", llm_ms, status="error")
            logger.warning("llm_completion_failed error_type=%s", type(exc).__name__)
            sources = self._sources_from_chunks(chunks)
            debug_payload = self._debug_payload(
                debug=debug,
                trace=trace,
                decision=decision,
                timings_ms={
                    "retrieval": retrieval_ms,
                    "llm": llm_ms,
                    "total": total_ms,
                },
            )
            if debug_payload is not None:
                debug_payload["error"] = {"code": "llm_unavailable", "stage": "llm"}
            recorded_session_id = self._record_audit(
                question=question,
                answer=LLM_UNAVAILABLE_ANSWER,
                sources=sources,
                context=context,
                refusal_reason="llm_unavailable",
                latency_ms=total_ms,
                session_id=session_id,
            )
            return Answer(
                answer=LLM_UNAVAILABLE_ANSWER,
                sources=sources,
                debug=debug_payload,
                session_id=recorded_session_id,
                refusal_reason="llm_unavailable",
            )
        metrics.observe("rag_llm_latency_ms", llm_ms, status="success")
        total_ms = elapsed_ms(started_at)
        debug_payload = self._debug_payload(
            debug=debug,
            trace=trace,
            decision=decision,
            timings_ms={
                "retrieval": retrieval_ms,
                "llm": llm_ms,
                "total": total_ms,
            },
        )
        sources = self._sources_from_chunks(chunks)
        recorded_session_id = self._record_audit(
            question=question,
            answer=answer,
            sources=sources,
            context=context,
            refusal_reason=None,
            latency_ms=total_ms,
            session_id=session_id,
        )
        return Answer(
            answer=answer,
            sources=sources,
            debug=debug_payload,
            session_id=recorded_session_id,
        )

    def answer_stream(
        self,
        question: str,
        top_k: int,
        *,
        debug: bool = False,
        retrieval_options: RetrievalDebugOptions | None = None,
        context: RequestContext | None = None,
        session_id: str | None = None,
    ):
        started_at = time.perf_counter()
        retrieval_started_at = started_at
        chunks, trace = self._retrieve(
            question,
            top_k=top_k,
            debug=debug,
            retrieval_options=retrieval_options,
            context=context,
        )
        retrieval_ms = elapsed_ms(retrieval_started_at)
        metrics.observe("rag_retrieval_latency_ms", retrieval_ms)
        decision = decide_retrieval_confidence(
            question,
            chunks,
            trace=trace,
            config=self.confidence_config,
        )
        if decision.should_refuse:
            yield {"event": "token", "data": REFUSAL_ANSWER}
            yield {"event": "sources", "data": "[]"}
            total_ms = elapsed_ms(started_at)
            recorded_session_id = self._record_audit(
                question=question,
                answer=REFUSAL_ANSWER,
                sources=[],
                context=context,
                refusal_reason=decision.refusal_reason,
                latency_ms=total_ms,
                session_id=session_id,
            )
            if recorded_session_id is not None:
                yield {"event": "session", "data": json.dumps({"session_id": recorded_session_id})}
            debug_payload = self._debug_payload(
                debug=debug,
                trace=trace,
                decision=decision,
                timings_ms={"retrieval": retrieval_ms, "llm": 0.0, "total": total_ms},
            )
            if debug_payload is not None:
                yield {"event": "debug", "data": json.dumps(debug_payload, ensure_ascii=False)}
            return

        prompt = build_rag_prompt(question, chunks)
        llm_started_at = time.perf_counter()
        llm_error = None
        answer_parts = []
        try:
            stream = self.llm.stream(prompt) if hasattr(self.llm, "stream") else [self.llm.complete(prompt)]
            for token in stream:
                answer_parts.append(str(token))
                yield {"event": "token", "data": token}
        except Exception as exc:
            llm_error = exc
            metrics.increment("rag_errors_total", stage="llm")
            logger.warning("llm_stream_failed error_type=%s", type(exc).__name__)
            answer_parts = [LLM_UNAVAILABLE_ANSWER]
            yield {"event": "token", "data": LLM_UNAVAILABLE_ANSWER}
        llm_ms = elapsed_ms(llm_started_at)
        metrics.observe("rag_llm_latency_ms", llm_ms, status="error" if llm_error else "success")

        source_objects = self._sources_from_chunks(chunks)
        sources = [
            {
                "source": source.source,
                "page": source.page,
                "chunk_index": source.chunk_index,
                "matched_child_chunk_index": source.matched_child_chunk_index,
                "content_type": source.content_type,
                "table_index": source.table_index,
                "content": source.content,
            }
            for source in source_objects
        ]
        yield {"event": "sources", "data": json.dumps(sources, ensure_ascii=False)}
        total_ms = elapsed_ms(started_at)
        recorded_session_id = self._record_audit(
            question=question,
            answer="".join(answer_parts),
            sources=source_objects,
            context=context,
            refusal_reason="llm_unavailable" if llm_error else None,
            latency_ms=total_ms,
            session_id=session_id,
        )
        if recorded_session_id is not None:
            yield {"event": "session", "data": json.dumps({"session_id": recorded_session_id})}
        debug_payload = self._debug_payload(
            debug=debug,
            trace=trace,
            decision=decision,
            timings_ms={
                "retrieval": retrieval_ms,
                "llm": llm_ms,
                "total": total_ms,
            },
        )
        if debug_payload is not None:
            if llm_error is not None:
                debug_payload["error"] = {"code": "llm_unavailable", "stage": "llm"}
            yield {"event": "debug", "data": json.dumps(debug_payload, ensure_ascii=False)}

    def _sources_from_chunks(self, chunks: list[Chunk]) -> list[Source]:
        return [
            Source(
                source=chunk.source,
                page=chunk.metadata.get("page"),
                chunk_index=chunk.metadata.get("chunk_index"),
                content=chunk.content,
                matched_child_chunk_index=chunk.metadata.get("matched_child_chunk_index"),
                content_type=chunk.metadata.get("content_type"),
                table_index=chunk.metadata.get("table_index"),
                metadata=dict(chunk.metadata),
            )
            for chunk in chunks
        ]

    def _retrieve(
        self,
        question: str,
        *,
        top_k: int,
        debug: bool,
        retrieval_options: RetrievalDebugOptions | None,
        context: RequestContext | None,
    ) -> tuple[list[Chunk], RetrievalTrace | None]:
        if hasattr(self.retriever, "similarity_search_with_trace"):
            kwargs = self._retrieval_context_kwargs(
                self.retriever.similarity_search_with_trace,
                context=context,
            )
            if self._method_accepts(self.retriever.similarity_search_with_trace, "options"):
                kwargs["options"] = retrieval_options
            result = self.retriever.similarity_search_with_trace(
                question,
                top_k=top_k,
                **kwargs,
            )
            return result.chunks, result.trace

        kwargs = self._retrieval_context_kwargs(self.retriever.similarity_search, context=context)
        return self.retriever.similarity_search(question, top_k=top_k, **kwargs), None

    def _retrieval_context_kwargs(self, method, *, context: RequestContext | None) -> dict:
        if context is None:
            return {}
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return {}
        if "context" in parameters:
            return {"context": context}
        if "access_filter" in parameters:
            return {"access_filter": RetrievalAccessFilter.from_context(context)}
        return {}

    def _method_accepts(self, method, parameter: str) -> bool:
        try:
            return parameter in inspect.signature(method).parameters
        except (TypeError, ValueError):
            return False

    def _debug_payload(
        self,
        *,
        debug: bool,
        trace: RetrievalTrace | None,
        decision: RetrievalConfidenceDecision,
        timings_ms: dict,
    ) -> dict | None:
        if not debug:
            return None
        payload = trace.to_dict() if trace is not None else {}
        payload["confidence"] = decision.to_dict()
        payload["timings_ms"] = timings_ms
        return payload

    def _record_audit(
        self,
        *,
        question: str,
        answer: str,
        sources: list[Source],
        context: RequestContext | None,
        refusal_reason: str | None,
        latency_ms: float,
        session_id: str | None,
    ) -> str | None:
        if self.audit_repository is None:
            return session_id
        try:
            return self.audit_repository.record_qa_session(
                question=question,
                answer=answer,
                sources=sources,
                context=context,
                refusal_reason=refusal_reason,
                latency_ms=latency_ms,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning("qa_audit_write_failed error=%s", exc)
            return session_id


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)
