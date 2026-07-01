from app.core.config import get_settings
from app.ingestion.index_versions import activate_index_version, get_index_paths, validate_index_build, versioned_indexing_enabled
from app.ingestion.pipeline import ingest_directory
from app.rag.embeddings import build_embeddings
from app.rag.vector_store import ChromaVectorStore


def main() -> None:
    settings = get_settings()
    index_paths = get_index_paths(settings)
    embeddings = build_embeddings(settings)
    vector_store = ChromaVectorStore(
        persist_dir=index_paths.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=embeddings,
    )
    result = ingest_directory(
        documents_dir=settings.documents_dir,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        reset=True,
        bm25_corpus_path=index_paths.bm25_corpus_path,
        parent_corpus_path=index_paths.parent_corpus_path,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
        manifest_path=settings.documents_manifest_path,
        default_tenant_id=settings.default_tenant_id,
        document_index_version=index_paths.version,
    )
    if versioned_indexing_enabled(settings):
        validate_index_build(index_paths, result)
        activate_index_version(settings, index_paths.version, require_exists=True)
    print(result)


if __name__ == "__main__":
    main()
