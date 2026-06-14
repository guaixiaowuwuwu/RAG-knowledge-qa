from app.core.config import get_settings
from app.ingestion.pipeline import ingest_directory
from app.rag.embeddings import build_embeddings
from app.rag.vector_store import ChromaVectorStore


def main() -> None:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=embeddings,
    )
    result = ingest_directory(
        documents_dir=settings.documents_dir,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        reset=True,
        bm25_corpus_path=settings.bm25_corpus_path,
    )
    print(result)


if __name__ == "__main__":
    main()
