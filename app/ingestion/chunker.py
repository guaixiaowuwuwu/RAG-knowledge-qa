from dataclasses import dataclass
from hashlib import sha1

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loaders import LoadedDocument


@dataclass(frozen=True)
class Chunk:
    content: str
    source: str
    metadata: dict


@dataclass(frozen=True)
class ParentChildChunks:
    parents: list[Chunk]
    children: list[Chunk]


def parent_id_for(source: str, parent_index: int, content: str) -> str:
    raw = f"{source}:parent:{parent_index}:{content}"
    return sha1(raw.encode("utf-8")).hexdigest()


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks: list[Chunk] = []
    for document in documents:
        if not document.text.strip():
            continue

        parts = _split_document_text(document, splitter, chunk_size)
        for index, part in enumerate(parts):
            content = part.strip()
            if not content:
                continue
            metadata = dict(document.metadata)
            metadata["source"] = document.source
            metadata["chunk_index"] = index
            chunks.append(Chunk(content=content, source=document.source, metadata=metadata))

    return chunks


def chunk_documents_with_parents(
    documents: list[LoadedDocument],
    child_chunk_size: int,
    child_chunk_overlap: int,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
) -> ParentChildChunks:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for document in documents:
        if not document.text.strip():
            continue

        parent_parts = _split_document_text(document, parent_splitter, parent_chunk_size)
        for parent_index, parent_part in enumerate(parent_parts):
            parent_content = parent_part.strip()
            if not parent_content:
                continue

            identity = parent_id_for(document.source, parent_index, parent_content)
            parent_metadata = dict(document.metadata)
            parent_metadata["source"] = document.source
            parent_metadata["chunk_index"] = parent_index
            parent_metadata["parent_id"] = identity
            parents.append(Chunk(content=parent_content, source=document.source, metadata=parent_metadata))

            child_document = LoadedDocument(text=parent_content, source=document.source, metadata=document.metadata)
            child_parts = _split_document_text(child_document, child_splitter, child_chunk_size)
            for child_index, child_part in enumerate(child_parts):
                child_content = child_part.strip()
                if not child_content:
                    continue
                child_metadata = dict(document.metadata)
                child_metadata["source"] = document.source
                child_metadata["chunk_index"] = child_index
                child_metadata["parent_id"] = identity
                children.append(Chunk(content=child_content, source=document.source, metadata=child_metadata))

    return ParentChildChunks(parents=parents, children=children)


def _split_document_text(document: LoadedDocument, splitter: RecursiveCharacterTextSplitter, chunk_size: int) -> list[str]:
    if document.metadata.get("content_type") == "table":
        return split_markdown_table(document.text, chunk_size=chunk_size)
    return splitter.split_text(document.text)


def split_markdown_table(markdown: str, chunk_size: int) -> list[str]:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) <= 2 or len(markdown) <= chunk_size:
        return [markdown]

    header = lines[:2]
    rows = lines[2:]
    chunks: list[str] = []
    current = header.copy()

    for row in rows:
        candidate = current + [row]
        if len("\n".join(candidate)) > chunk_size and len(current) > len(header):
            chunks.append("\n".join(current))
            current = header + [row]
        else:
            current = candidate

    if len(current) > len(header):
        chunks.append("\n".join(current))

    return chunks or [markdown]
