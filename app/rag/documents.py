from dataclasses import dataclass
from hashlib import sha1

from app.ingestion.chunker import Chunk


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    content: str
    source: str
    metadata: dict
    score: float | None = None


def chunk_id(chunk: Chunk) -> str:
    content_type = chunk.metadata.get("content_type", "")
    parent_id = chunk.metadata.get("parent_id", "")
    chunk_index = chunk.metadata.get("chunk_index", "")
    page = chunk.metadata.get("page", "")
    table_index = chunk.metadata.get("table_index", "")
    raw = f"{chunk.source}:{content_type}:{parent_id}:{page}:{table_index}:{chunk_index}:{chunk.content}"
    return sha1(raw.encode("utf-8")).hexdigest()


def chunk_to_retrieved_document(chunk: Chunk, score: float | None = None) -> RetrievedDocument:
    metadata = dict(chunk.metadata)
    identity = str(metadata.get("chunk_id") or chunk_id(chunk))
    metadata["chunk_id"] = identity
    metadata["source"] = chunk.source
    return RetrievedDocument(
        id=identity,
        content=chunk.content,
        source=chunk.source,
        metadata=metadata,
        score=score,
    )


def retrieved_document_to_chunk(document: RetrievedDocument) -> Chunk:
    metadata = dict(document.metadata)
    metadata["chunk_id"] = document.id
    metadata["source"] = document.source
    return Chunk(content=document.content, source=document.source, metadata=metadata)
