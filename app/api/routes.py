import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    AuditFeedbackResponse,
    AuditSessionDetailResponse,
    AuditSessionListResponse,
    AuditSessionResponse,
    AuditSourceResponse,
    AnswerEvaluationReportResponse,
    AskRequest,
    AskResponse,
    CorpusStatusResponse,
    EvaluationReportResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    IndexActivationResponse,
    IndexVersionListResponse,
    IndexVersionResponse,
    IngestionJobCreateRequest,
    IngestionJobResponse,
    RetrievalComparisonReportResponse,
    SourceResponse,
)
from app.audit.models import AuditFeedback, AuditSession, AuditSessionDetail, AuditSource
from app.audit.repository import build_audit_repository
from app.core.config import get_settings
from app.evaluation.dataset import load_eval_cases
from app.evaluation.report import build_retrieval_report
from app.ingestion.index_versions import (
    activate_index_version,
    get_active_index_version,
    get_index_paths,
    list_index_versions,
    validate_index_build,
    versioned_indexing_enabled,
)
from app.ingestion.jobs import IngestionJob, IngestionJobRepository
from app.ingestion.pipeline import ingest_directory
from app.rag.bm25 import BM25Retriever
from app.rag.cache import build_answer_cache, build_cache_key, should_use_answer_cache
from app.rag.confidence import RetrievalConfidenceConfig, decide_retrieval_confidence
from app.rag.documents import chunk_to_retrieved_document
from app.rag.embeddings import build_embeddings
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.llm import OpenAIChatLLM
from app.rag.parent_store import JsonlParentStore
from app.rag.query_transform import QueryTransformer
from app.rag.reranker import build_bge_reranker
from app.rag.hybrid_retriever import RetrievalDebugOptions
from app.rag.service import RagService
from app.rag.vector_store import ChromaVectorStore
from app.observability.metrics import metrics, record_ingestion_job_status
from app.security.auth import require_admin, require_authenticated
from app.security.context import RequestContext


router = APIRouter()
REPORTS_DIR = Path("reports")
logger = logging.getLogger(__name__)


def build_vector_store():
    settings = get_settings()
    index_paths = get_index_paths(settings)
    return ChromaVectorStore(
        persist_dir=index_paths.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=build_embeddings(settings),
    )


def build_rag_service():
    settings = get_settings()
    llm = OpenAIChatLLM(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
        timeout_seconds=getattr(settings, "llm_timeout_seconds", 60.0),
    )
    return RagService(
        retriever=build_retriever(),
        llm=llm,
        confidence_config=RetrievalConfidenceConfig.from_settings(settings),
        audit_repository=build_audit_repository(settings),
    )


def build_retriever():
    settings = get_settings()
    index_paths = get_index_paths(settings)
    dense = build_vector_store()
    sparse = BM25Retriever.from_jsonl(index_paths.bm25_corpus_path)
    reranker = build_bge_reranker(settings.reranker_model)
    parent_store = JsonlParentStore(index_paths.parent_corpus_path)
    transformer_llm = OpenAIChatLLM(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
        timeout_seconds=getattr(settings, "llm_timeout_seconds", 60.0),
    )
    query_transformer = QueryTransformer(
        llm=transformer_llm,
        rewrite_enabled=settings.query_rewrite_enabled,
        hyde_enabled=settings.hyde_enabled,
        max_variants=settings.max_query_variants,
        timeout_seconds=settings.query_transform_timeout_seconds,
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
        permission_filter_overfetch_max=getattr(settings, "permission_filter_overfetch_max", 100),
    )


def build_evaluation_report():
    settings = get_settings()
    cases = load_eval_cases(settings.eval_dataset_path)
    retriever = build_retriever()

    report_cases = []
    confidence_config = RetrievalConfidenceConfig.from_settings(settings)
    for case in cases:
        chunks, trace = retrieve_chunks_with_trace(retriever, case.question, top_k=settings.retrieval_top_k)
        decision = decide_retrieval_confidence(
            case.question,
            chunks,
            trace=trace,
            config=confidence_config,
        )
        if decision.should_refuse:
            chunks = []
        report_cases.append(
            {
                "id": case.id,
                "question": case.question,
                "ground_truth": case.ground_truth,
                "expected_sources": case.expected_sources,
                "expected_answer_keywords": case.expected_answer_keywords,
                "expected_pages": case.expected_pages,
                "expected_chunk_keywords": case.expected_chunk_keywords,
                "evidence_notes": case.evidence_notes,
                "category": case.category,
                "difficulty": case.difficulty,
                "language": case.language,
                "is_negative": case.is_negative,
                "retrieved": [chunk_to_retrieved_document(chunk) for chunk in chunks],
                "refusal_reason": decision.refusal_reason,
                "confidence": decision.to_dict(),
            }
        )
    return build_retrieval_report(
        report_cases,
        dataset_path=str(settings.eval_dataset_path),
        variant="full",
        top_k=settings.retrieval_top_k,
    )


def build_latest_answer_evaluation_report() -> dict:
    report_paths = sorted(
        (
            path
            for path in REPORTS_DIR.glob("answer-eval*.json")
            if not path.name.startswith("answer-eval-failed")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not report_paths:
        return {
            "available": False,
            "error": "No answer evaluation report found. Run `python -m scripts.evaluate_answers --limit 10` first.",
        }

    report_path = report_paths[0]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "report_path": str(report_path),
            "error": f"Could not read answer evaluation report: {type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "report_path": str(report_path),
        "generated_at": report.get("generated_at") or report.get("config", {}).get("answer_eval_generated_at"),
        "dataset_path": report.get("dataset_path"),
        "top_k": report.get("top_k"),
        "summary": report.get("summary", {}),
        "ragas": report.get("ragas"),
        "error": report.get("error") or ("Latest answer evaluation report does not contain RAGAS metrics." if not report.get("ragas") else None),
    }


def build_latest_retrieval_comparison_report() -> dict:
    report_paths = sorted(
        REPORTS_DIR.glob("retrieval-comparison*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not report_paths:
        return {
            "available": False,
            "error": "No retrieval comparison report found. Run `python -m scripts.evaluate --compare --top-k 5` first.",
        }

    report_path = report_paths[0]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "report_path": str(report_path),
            "error": f"Could not read retrieval comparison report: {type(exc).__name__}: {exc}",
        }

    variants = []
    for variant in report.get("variants", []):
        variants.append(
            {
                "variant": variant.get("variant"),
                "generated_at": variant.get("generated_at"),
                "summary": variant.get("summary", {}),
                "groups": variant.get("groups", {}),
            }
        )

    return {
        "available": True,
        "report_path": str(report_path),
        "generated_at": report.get("generated_at"),
        "dataset_path": report.get("dataset_path"),
        "top_k": report.get("top_k"),
        "variants": variants,
        "table": report.get("table"),
    }


def retrieve_chunks_with_trace(retriever, question: str, *, top_k: int):
    if hasattr(retriever, "similarity_search_with_trace"):
        result = retriever.similarity_search_with_trace(question, top_k=top_k)
        return result.chunks, result.trace
    return retriever.similarity_search(question, top_k=top_k), None


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint():
    return PlainTextResponse(metrics.render_prometheus(), media_type="text/plain; version=0.0.4")


@router.get("/evaluation/report", response_model=EvaluationReportResponse)
def evaluation_report(_context: RequestContext = Depends(require_admin)):
    return build_evaluation_report()


@router.get("/evaluation/answer-report", response_model=AnswerEvaluationReportResponse)
def answer_evaluation_report(_context: RequestContext = Depends(require_admin)):
    return build_latest_answer_evaluation_report()


@router.get("/evaluation/comparison-report", response_model=RetrievalComparisonReportResponse)
def retrieval_comparison_report(_context: RequestContext = Depends(require_admin)):
    return build_latest_retrieval_comparison_report()


@router.get("/corpus/status", response_model=CorpusStatusResponse)
def corpus_status(_context: RequestContext = Depends(require_admin)):
    return build_corpus_status()


@router.post("/ingest", response_model=IngestResponse)
def ingest(_context: RequestContext = Depends(require_admin)):
    settings = get_settings()
    ingestion_mode = str(getattr(settings, "ingestion_mode", "sync")).lower()
    if ingestion_mode == "async":
        repository = IngestionJobRepository.from_settings(settings)
        job = repository.create_job(
            tenant_id=_context.tenant_id,
            requested_by=_context.user_id,
            input_path=settings.documents_dir,
            target_index_version=None,
        )
        record_ingestion_job_status(job.status)
        return IngestResponse(
            loaded_documents=0,
            indexed_chunks=0,
            skipped=[],
            errors={},
            job_id=job.id,
            status=job.status,
            target_index_version=job.target_index_version,
        )

    index_paths = get_index_paths(settings)
    result = ingest_directory(
        documents_dir=settings.documents_dir,
        vector_store=build_vector_store(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        reset=True,
        bm25_corpus_path=index_paths.bm25_corpus_path,
        parent_corpus_path=index_paths.parent_corpus_path,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
        manifest_path=getattr(settings, "documents_manifest_path", None),
        default_tenant_id=getattr(settings, "default_tenant_id", "default"),
        document_index_version=index_paths.version,
    )
    if versioned_indexing_enabled(settings):
        validate_index_build(index_paths, result)
        activate_index_version(settings, index_paths.version, require_exists=True)
    return IngestResponse(
        loaded_documents=result.loaded_documents,
        indexed_chunks=result.indexed_chunks,
        skipped=result.skipped,
        errors=result.errors,
    )


@router.post("/admin/ingestion/jobs", response_model=IngestionJobResponse)
def create_ingestion_job(request: IngestionJobCreateRequest, context: RequestContext = Depends(require_admin)):
    settings = get_settings()
    repository = IngestionJobRepository.from_settings(settings)
    job = repository.create_job(
        tenant_id=context.tenant_id,
        requested_by=context.user_id,
        input_path=request.input_path or str(settings.documents_dir),
        target_index_version=request.target_index_version,
    )
    record_ingestion_job_status(job.status)
    return ingestion_job_response(job)


@router.get("/admin/ingestion/jobs/{job_id}", response_model=IngestionJobResponse)
def ingestion_job_detail(job_id: str, context: RequestContext = Depends(require_admin)):
    repository = IngestionJobRepository.from_settings(get_settings())
    job = repository.get_job(job_id, tenant_id=context.tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return ingestion_job_response(job)


@router.post("/admin/indexes/{version}/activate", response_model=IndexActivationResponse)
def activate_index(version: str, _context: RequestContext = Depends(require_admin)):
    settings = get_settings()
    try:
        activate_index_version(settings, version, require_exists=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IndexActivationResponse(version=version, active_version=get_active_index_version(settings))


@router.get("/admin/indexes", response_model=IndexVersionListResponse)
def indexes(context: RequestContext = Depends(require_admin)):
    settings = get_settings()
    repository = IngestionJobRepository.from_settings(settings)
    statuses = repository.latest_status_by_version(tenant_id=context.tenant_id)
    return IndexVersionListResponse(
        active_version=get_active_index_version(settings),
        indexes=[
            IndexVersionResponse(
                version=index.version,
                index_dir=str(index.index_dir),
                is_active=index.is_active,
                exists=index.exists,
                bm25_count=index.bm25_count,
                parent_count=index.parent_count,
                chroma_exists=index.chroma_exists,
                updated_at=index.updated_at,
                job_status=statuses.get(index.version),
            )
            for index in list_index_versions(settings)
        ],
    )


@router.get("/admin/audit/sessions", response_model=AuditSessionListResponse)
def audit_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    context: RequestContext = Depends(require_admin),
):
    repository = build_audit_repository(get_settings())
    return AuditSessionListResponse(
        sessions=[
            audit_session_response(session)
            for session in repository.list_sessions(
                tenant_id=context.tenant_id,
                limit=limit,
                offset=offset,
            )
        ]
    )


@router.get("/admin/audit/sessions/{session_id}", response_model=AuditSessionDetailResponse)
def audit_session_detail(session_id: str, context: RequestContext = Depends(require_admin)):
    repository = build_audit_repository(get_settings())
    detail = repository.get_session(session_id, tenant_id=context.tenant_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Audit session not found.")
    return audit_session_detail_response(detail)


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest, context: RequestContext = Depends(require_authenticated)):
    repository = build_audit_repository(get_settings())
    feedback_id = repository.add_feedback(
        session_id=request.session_id,
        context=context,
        rating=request.rating,
        tags=request.tags,
        comment=request.comment,
    )
    if feedback_id is None:
        raise HTTPException(status_code=404, detail="Audit session not found.")
    return FeedbackResponse(feedback_id=feedback_id, session_id=request.session_id)


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, context: RequestContext = Depends(require_authenticated)):
    settings = get_settings()
    validate_question_length(request.question, settings.max_question_chars)
    top_k = request.top_k or settings.retrieval_top_k
    retrieval_options = retrieval_options_from_request(request)
    cache_key = build_cache_key(
        question=request.question,
        top_k=top_k,
        settings=settings,
        retrieval_options=retrieval_options,
        context=context,
    )
    cache = build_answer_cache(settings)
    cache_enabled = should_use_answer_cache(settings, context=context)
    if cache_enabled:
        try:
            cached = cache.get(cache_key)
        except Exception as exc:
            logger.warning("answer_cache_read_failed key=%s error=%s", cache_key, exc)
            cached = None
        if cached is not None:
            metrics.increment("rag_cache_hits_total")
            response = response_from_cached_payload(cached, debug=request.debug, cache_key=cache_key)
            response.session_id = record_cached_answer_audit(
                settings=settings,
                question=request.question,
                response=response,
                context=context,
            )
            return response
        metrics.increment("rag_cache_misses_total")

    service = build_rag_service()
    try:
        answer = service.answer(
            question=request.question,
            top_k=top_k,
            debug=request.debug,
            retrieval_options=retrieval_options,
            context=context,
        )
    except Exception as exc:
        metrics.increment("rag_errors_total", stage="api")
        logger.exception("rag_answer_failed request_id_unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "rag_unavailable", "message": "问答服务暂时不可用，请稍后重试。"},
        ) from exc
    refusal_reason = getattr(answer, "refusal_reason", None)
    if refusal_reason:
        metrics.increment("rag_refusals_total", reason=refusal_reason)
    if not answer.sources:
        metrics.increment("rag_empty_retrievals_total")
    response = AskResponse(
        answer=answer.answer,
        sources=[
            SourceResponse(
                source=source.source,
                page=source.page,
                chunk_index=source.chunk_index,
                matched_child_chunk_index=source.matched_child_chunk_index,
                content_type=source.content_type,
                table_index=source.table_index,
                content=source.content,
            )
            for source in answer.sources
        ],
        debug=answer.debug,
        session_id=getattr(answer, "session_id", None),
    )
    if cache_enabled:
        cache_payload = {
            "answer": response.answer,
            "sources": [source.model_dump() for source in response.sources],
        }
        try:
            cache.set(cache_key, cache_payload)
        except Exception as exc:
            logger.warning("answer_cache_write_failed key=%s error=%s", cache_key, exc)
        if request.debug:
            response.debug = dict(response.debug or {})
            response.debug["cache"] = {"hit": False, "key": cache_key}
    return response


@router.post("/ask/stream")
def ask_stream(request: AskRequest, context: RequestContext = Depends(require_authenticated)):
    settings = get_settings()
    validate_question_length(request.question, settings.max_question_chars)
    service = build_rag_service()
    events = service.answer_stream(
        question=request.question,
        top_k=request.top_k or settings.retrieval_top_k,
        debug=request.debug,
        retrieval_options=retrieval_options_from_request(request),
        context=context,
    )
    return EventSourceResponse(events)


def retrieval_options_from_request(request: AskRequest) -> RetrievalDebugOptions:
    return RetrievalDebugOptions(
        rewrite_enabled=request.rewrite_enabled,
        hyde_enabled=request.hyde_enabled,
        parent_hydration_enabled=request.parent_hydration_enabled,
        max_query_variants=request.max_query_variants,
    )


def validate_question_length(question: str, max_question_chars: int) -> None:
    if len(question) > max_question_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Question exceeds MAX_QUESTION_CHARS/max_question_chars limit ({max_question_chars}).",
        )


def response_from_cached_payload(payload: dict, *, debug: bool, cache_key: str) -> AskResponse:
    debug_payload = None
    if debug:
        debug_payload = dict(payload.get("debug") or {})
        debug_payload["cache"] = {"hit": True, "key": cache_key}
    return AskResponse(
        answer=str(payload.get("answer", "")),
        sources=[SourceResponse(**source) for source in payload.get("sources", [])],
        debug=debug_payload,
    )


def record_cached_answer_audit(
    *,
    settings,
    question: str,
    response: AskResponse,
    context: RequestContext,
) -> str | None:
    started_at = time.perf_counter()
    try:
        return build_audit_repository(settings).record_qa_session(
            question=question,
            answer=response.answer,
            sources=response.sources,
            context=context,
            refusal_reason=None,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )
    except Exception as exc:
        logger.warning("qa_audit_cache_hit_write_failed error=%s", exc)
        return None


def audit_session_response(session: AuditSession) -> AuditSessionResponse:
    return AuditSessionResponse(
        id=session.id,
        tenant_id=session.tenant_id,
        user_id_hash=session.user_id_hash,
        source=session.source,
        index_version=session.index_version,
        question_hash=session.question_hash,
        redacted_question=session.redacted_question,
        answer_summary=session.answer_summary,
        refusal_reason=session.refusal_reason,
        latency_ms=session.latency_ms,
        token_usage=session.token_usage,
        created_at=session.created_at,
    )


def ingestion_job_response(job: IngestionJob) -> IngestionJobResponse:
    return IngestionJobResponse(
        id=job.id,
        tenant_id=job.tenant_id,
        requested_by=job.requested_by,
        status=job.status,
        input_path=job.input_path,
        target_index_version=job.target_index_version,
        error=job.error,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def audit_source_response(source: AuditSource) -> AuditSourceResponse:
    return AuditSourceResponse(
        source_path=source.source_path,
        page=source.page,
        chunk_id=source.chunk_id,
        chunk_index=source.chunk_index,
        document_version=source.document_version,
    )


def audit_feedback_response(feedback: AuditFeedback) -> AuditFeedbackResponse:
    return AuditFeedbackResponse(
        id=feedback.id,
        session_id=feedback.session_id,
        tenant_id=feedback.tenant_id,
        user_id_hash=feedback.user_id_hash,
        rating=feedback.rating,
        tags=list(feedback.tags),
        comment=feedback.comment,
        created_at=feedback.created_at,
    )


def audit_session_detail_response(detail: AuditSessionDetail) -> AuditSessionDetailResponse:
    return AuditSessionDetailResponse(
        session=audit_session_response(detail.session),
        sources=[audit_source_response(source) for source in detail.sources],
        feedback=[audit_feedback_response(feedback) for feedback in detail.feedback],
    )


def build_corpus_status() -> dict:
    settings = get_settings()
    index_paths = get_index_paths(settings)
    bm25_status = jsonl_status(index_paths.bm25_corpus_path)
    parent_status = jsonl_status(index_paths.parent_corpus_path)
    chroma_status = chroma_collection_status(index_paths.chroma_dir, settings.chroma_collection)
    ready, readiness_reason = corpus_readiness(bm25_status, parent_status, chroma_status)

    return {
        "documents_dir": str(settings.documents_dir),
        "active_index_version": index_paths.version,
        "index_dir": str(index_paths.index_dir),
        "ready": ready,
        "readiness_reason": readiness_reason,
        "document_count": count_document_files(settings.documents_dir),
        "chunk_count": bm25_status["count"] or chroma_status["chunk_count"],
        "parent_chunk_count": parent_status["count"],
        "bm25_ready": bm25_status["exists"] and bm25_status["count"] > 0,
        "chroma_collection_name": settings.chroma_collection,
        "bm25_corpus": bm25_status,
        "parent_corpus": parent_status,
        "chroma": chroma_status,
    }


def corpus_readiness(bm25_status: dict, parent_status: dict, chroma_status: dict) -> tuple[bool, str]:
    if int(chroma_status.get("chunk_count") or 0) <= 0:
        return False, "missing_chroma_chunks"
    if not bm25_status.get("exists") or int(bm25_status.get("count") or 0) <= 0:
        return False, "missing_bm25_corpus"
    if not parent_status.get("exists") or int(parent_status.get("count") or 0) <= 0:
        return False, "missing_parent_corpus"
    return True, "ready"


def count_document_files(directory: Path) -> int:
    if not directory.exists():
        return 0

    supported_suffixes = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def jsonl_status(path: Path) -> dict:
    exists = path.exists()
    count = 0
    size_bytes = 0
    updated_at = None

    if exists:
        stat = path.stat()
        size_bytes = stat.st_size
        updated_at = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
        with path.open("r", encoding="utf-8") as file:
            count = sum(1 for line in file if line.strip())

    return {
        "path": str(path),
        "exists": exists,
        "count": count,
        "size_bytes": size_bytes,
        "updated_at": updated_at,
    }


def chroma_collection_status(persist_dir: Path, collection_name: str) -> dict:
    sqlite_path = persist_dir / "chroma.sqlite3"
    exists = sqlite_path.exists()
    size_bytes = directory_size(persist_dir) if persist_dir.exists() else 0
    updated_at = latest_modified_at(persist_dir) if persist_dir.exists() else None
    chunk_count = 0
    error = None

    if exists:
        try:
            with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM embeddings
                    JOIN segments ON embeddings.segment_id = segments.id
                    JOIN collections ON segments.collection = collections.id
                    WHERE collections.name = ?
                    """,
                    (collection_name,),
                ).fetchone()
                chunk_count = int(row[0] or 0) if row else 0
        except sqlite3.Error as exc:
            error = f"{type(exc).__name__}: {exc}"

    return {
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "exists": exists,
        "chunk_count": chunk_count,
        "size_bytes": size_bytes,
        "updated_at": updated_at,
        "error": error,
    }


def directory_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def latest_modified_at(directory: Path) -> str | None:
    modified_times = [path.stat().st_mtime for path in directory.rglob("*") if path.exists()]
    if not modified_times:
        return None
    return datetime.fromtimestamp(max(modified_times), UTC).isoformat()
