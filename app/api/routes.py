from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import AskRequest, AskResponse, EvaluationReportResponse, IngestResponse, SourceResponse
from app.core.config import get_settings
from app.evaluation.dataset import load_eval_cases
from app.evaluation.report import build_retrieval_report
from app.ingestion.pipeline import ingest_directory
from app.rag.bm25 import BM25Retriever
from app.rag.documents import chunk_to_retrieved_document
from app.rag.embeddings import build_embeddings
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.llm import OpenAIChatLLM
from app.rag.parent_store import JsonlParentStore
from app.rag.query_transform import QueryTransformer
from app.rag.reranker import build_bge_reranker
from app.rag.service import RagService
from app.rag.vector_store import ChromaVectorStore


router = APIRouter()


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
    )
    query_transformer = QueryTransformer(
        llm=transformer_llm,
        rewrite_enabled=settings.query_rewrite_enabled,
        hyde_enabled=settings.hyde_enabled,
        max_variants=settings.max_query_variants,
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
                "question": case.question,
                "expected_sources": case.expected_sources,
                "retrieved": [chunk_to_retrieved_document(chunk) for chunk in chunks],
            }
        )
    return build_retrieval_report(report_cases)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/evaluation/report", response_model=EvaluationReportResponse)
def evaluation_report():
    return build_evaluation_report()


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
    service = build_rag_service()
    answer = service.answer(
        question=request.question,
        top_k=request.top_k or settings.retrieval_top_k,
    )
    return AskResponse(
        answer=answer.answer,
        sources=[
            SourceResponse(
                source=source.source,
                page=source.page,
                chunk_index=source.chunk_index,
                content=source.content,
            )
            for source in answer.sources
        ],
    )


@router.post("/ask/stream")
def ask_stream(request: AskRequest):
    settings = get_settings()
    service = build_rag_service()
    events = service.answer_stream(
        question=request.question,
        top_k=request.top_k or settings.retrieval_top_k,
    )
    return EventSourceResponse(events)
