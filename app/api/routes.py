from fastapi import APIRouter

from app.api.schemas import AskRequest, AskResponse, IngestResponse, SourceResponse
from app.core.config import get_settings
from app.ingestion.pipeline import ingest_directory
from app.rag.bm25 import BM25Retriever
from app.rag.embeddings import build_embeddings
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.llm import OpenAIChatLLM
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
    return HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
        dense_top_k=settings.dense_retrieval_top_k,
        sparse_top_k=settings.bm25_retrieval_top_k,
        rrf_k=settings.rrf_k,
        reranker_top_n=settings.reranker_top_n,
    )


@router.get("/health")
def health():
    return {"status": "ok"}


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
