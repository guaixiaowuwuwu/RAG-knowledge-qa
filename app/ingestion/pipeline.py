import json
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.loaders import load_documents_from_dir
from app.rag.documents import chunk_id


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
) -> IngestResult:
    load_result = load_documents_from_dir(documents_dir)
    chunks = chunk_documents(load_result.documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
