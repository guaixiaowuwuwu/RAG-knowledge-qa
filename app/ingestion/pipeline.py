import json
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunker import Chunk, chunk_documents, chunk_documents_with_parents
from app.ingestion.loaders import load_documents_from_dir
from app.ingestion.manifest import apply_document_manifest
from app.rag.documents import chunk_id
from app.rag.parent_store import JsonlParentStore


@dataclass(frozen=True)
class IngestResult:
    loaded_documents: int
    indexed_chunks: int
    skipped: list[str]
    errors: dict[str, str]


def ingest_directory(
    documents_dir: Path,
    vector_store,
    chunk_size: int,
    chunk_overlap: int,
    reset: bool = True,
    bm25_corpus_path: Path | None = None,
    parent_corpus_path: Path | None = None,
    parent_chunk_size: int | None = None,
    parent_chunk_overlap: int | None = None,
    manifest_path: Path | None = None,
    default_tenant_id: str = "default",
    document_index_version: str = "local-index-v1",
) -> IngestResult:
    load_result = load_documents_from_dir(documents_dir)
    documents = apply_document_manifest(
        load_result.documents,
        manifest_path=manifest_path,
        documents_dir=documents_dir,
        default_tenant_id=default_tenant_id,
        default_document_version=document_index_version,
    )
    if parent_corpus_path is not None and parent_chunk_size is not None and parent_chunk_overlap is not None:
        parent_child = chunk_documents_with_parents(
            documents,
            child_chunk_size=chunk_size,
            child_chunk_overlap=chunk_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
        )
        chunks = parent_child.children
        JsonlParentStore(parent_corpus_path).write(parent_child.parents)
    else:
        chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if reset:
        vector_store.reset()

    indexed = vector_store.add_chunks(chunks)
    if bm25_corpus_path is not None:
        persist_bm25_corpus(chunks, bm25_corpus_path)

    return IngestResult(
        loaded_documents=len(load_result.documents),
        indexed_chunks=indexed,
        skipped=load_result.skipped,
        errors=load_result.errors,
    )


def persist_bm25_corpus(chunks: list[Chunk], corpus_path: Path) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            identity = chunk_id(chunk)
            metadata = dict(chunk.metadata)
            metadata["chunk_id"] = identity
            metadata["source"] = chunk.source
            row = {
                "id": identity,
                "content": chunk.content,
                "source": chunk.source,
                "metadata": metadata,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
