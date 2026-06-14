from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import load_documents_from_dir


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
) -> IngestResult:
    load_result = load_documents_from_dir(documents_dir)
    chunks = chunk_documents(load_result.documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if reset:
        vector_store.reset()

    indexed = vector_store.add_chunks(chunks)
    return IngestResult(
        loaded_documents=len(load_result.documents),
        indexed_chunks=indexed,
        skipped=load_result.skipped,
        errors=load_result.errors,
    )
