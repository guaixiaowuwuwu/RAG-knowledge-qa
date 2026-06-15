import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    AnswerEvaluationReportResponse,
    AskRequest,
    AskResponse,
    ChromaCorpusStatusResponse,
    CorpusStatusResponse,
    EvaluationReportResponse,
    IngestResponse,
    JsonlCorpusStatusResponse,
    RetrievalComparisonReportResponse,
    SourceResponse,
)
from app.core.config import get_settings
from app.evaluation.dataset import load_eval_cases
from app.evaluation.report import build_retrieval_report
from app.ingestion.pipeline import ingest_directory
from app.rag.bm25 import BM25Retriever
from app.rag.cache import build_answer_cache, build_cache_key
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


router = APIRouter()
REPORTS_DIR = Path("reports")
logger = logging.getLogger(__name__)


def build_vector_store():
    settings = get_settings()
    return ChromaVectorStore(
        persist_dir=settings.chroma_dir,
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
    return RagService(retriever=build_retriever(), llm=llm)


def build_retriever():
    settings = get_settings()
    dense = build_vector_store()
    sparse = BM25Retriever.from_jsonl(settings.bm25_corpus_path)
    reranker = build_bge_reranker(settings.reranker_model)
    parent_store = JsonlParentStore(settings.parent_corpus_path)
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
    )


def build_evaluation_report():
    settings = get_settings()
    cases = load_eval_cases(settings.eval_dataset_path)
    retriever = build_retriever()

    report_cases = []
    for case in cases:
        chunks = retriever.similarity_search(case.question, top_k=settings.retrieval_top_k)
        report_cases.append(
            {
                "id": case.id,
                "question": case.question,
                "ground_truth": case.ground_truth,
                "expected_sources": case.expected_sources,
                "expected_answer_keywords": case.expected_answer_keywords,
                "is_negative": case.is_negative,
                "retrieved": [chunk_to_retrieved_document(chunk) for chunk in chunks],
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


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/evaluation/report", response_model=EvaluationReportResponse)
def evaluation_report():
    return build_evaluation_report()


@router.get("/evaluation/answer-report", response_model=AnswerEvaluationReportResponse)
def answer_evaluation_report():
    return build_latest_answer_evaluation_report()


@router.get("/evaluation/comparison-report", response_model=RetrievalComparisonReportResponse)
def retrieval_comparison_report():
    return build_latest_retrieval_comparison_report()


@router.get("/corpus/status", response_model=CorpusStatusResponse)
def corpus_status():
    return build_corpus_status()


@router.post("/ingest", response_model=IngestResponse)
def ingest():
    settings = get_settings()
    result = ingest_directory(
        documents_dir=settings.documents_dir,
        vector_store=build_vector_store(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        reset=True,
        bm25_corpus_path=settings.bm25_corpus_path,
        parent_corpus_path=settings.parent_corpus_path,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
    )
    return IngestResponse(
        loaded_documents=result.loaded_documents,
        indexed_chunks=result.indexed_chunks,
        skipped=result.skipped,
        errors=result.errors,
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    settings = get_settings()
    validate_question_length(request.question, settings.max_question_chars)
    top_k = request.top_k or settings.retrieval_top_k
    retrieval_options = retrieval_options_from_request(request)
    cache_key = build_cache_key(
        question=request.question,
        top_k=top_k,
        settings=settings,
        retrieval_options=retrieval_options,
    )
    cache = build_answer_cache(settings)
    if settings.answer_cache_enabled:
        try:
            cached = cache.get(cache_key)
        except Exception as exc:
            logger.warning("answer_cache_read_failed key=%s error=%s", cache_key, exc)
            cached = None
        if cached is not None:
            return response_from_cached_payload(cached, debug=request.debug, cache_key=cache_key)

    service = build_rag_service()
    answer = service.answer(
        question=request.question,
        top_k=top_k,
        debug=request.debug,
        retrieval_options=retrieval_options,
    )
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
    )
    if settings.answer_cache_enabled:
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
def ask_stream(request: AskRequest):
    settings = get_settings()
    validate_question_length(request.question, settings.max_question_chars)
    service = build_rag_service()
    events = service.answer_stream(
        question=request.question,
        top_k=request.top_k or settings.retrieval_top_k,
        debug=request.debug,
        retrieval_options=retrieval_options_from_request(request),
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


def build_corpus_status() -> dict:
    settings = get_settings()
    bm25_status = jsonl_status(settings.bm25_corpus_path)
    parent_status = jsonl_status(settings.parent_corpus_path)
    chroma_status = chroma_collection_status(settings.chroma_dir, settings.chroma_collection)

    return {
        "documents_dir": str(settings.documents_dir),
        "document_count": count_document_files(settings.documents_dir),
        "chunk_count": bm25_status.count or chroma_status.chunk_count,
        "parent_chunk_count": parent_status.count,
        "bm25_ready": bm25_status.exists and bm25_status.count > 0,
        "chroma_collection_name": settings.chroma_collection,
        "bm25_corpus": bm25_status,
        "parent_corpus": parent_status,
        "chroma": chroma_status,
    }


def count_document_files(directory: Path) -> int:
    if not directory.exists():
        return 0

    supported_suffixes = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def jsonl_status(path: Path) -> JsonlCorpusStatusResponse:
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

    return JsonlCorpusStatusResponse(
        path=str(path),
        exists=exists,
        count=count,
        size_bytes=size_bytes,
        updated_at=updated_at,
    )


def chroma_collection_status(persist_dir: Path, collection_name: str) -> ChromaCorpusStatusResponse:
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

    return ChromaCorpusStatusResponse(
        persist_dir=str(persist_dir),
        collection_name=collection_name,
        exists=exists,
        chunk_count=chunk_count,
        size_bytes=size_bytes,
        updated_at=updated_at,
        error=error,
    )


def directory_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def latest_modified_at(directory: Path) -> str | None:
    modified_times = [path.stat().st_mtime for path in directory.rglob("*") if path.exists()]
    if not modified_times:
        return None
    return datetime.fromtimestamp(max(modified_times), UTC).isoformat()
