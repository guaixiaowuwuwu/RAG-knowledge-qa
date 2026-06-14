from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loaders import LoadedDocument


@dataclass(frozen=True)
class Chunk:
    content: str
    source: str
    metadata: dict


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

        parts = splitter.split_text(document.text)
        for index, part in enumerate(parts):
            content = part.strip()
            if not content:
                continue
            metadata = dict(document.metadata)
            metadata["source"] = document.source
            metadata["chunk_index"] = index
            chunks.append(Chunk(content=content, source=document.source, metadata=metadata))

    return chunks
