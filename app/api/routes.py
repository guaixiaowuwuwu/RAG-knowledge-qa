from fastapi import APIRouter

from app.api.schemas import AskRequest, AskResponse, IngestResponse, SourceResponse
from app.core.config import get_settings
from app.ingestion.pipeline import ingest_directory
from app.rag.embeddings import OpenAIEmbeddings
from app.rag.llm import OpenAIChatLLM
from app.rag.service import RagService
from app.rag.vector_store import ChromaVectorStore


router = APIRouter()


def build_embeddings():
    settings = get_settings()
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )


def build_vector_store():
    settings = get_settings()
    return ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=build_embeddings(),
    )


def build_rag_service():
    settings = get_settings()
    llm = OpenAIChatLLM(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
    )
    return RagService(retriever=build_vector_store(), llm=llm)


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
