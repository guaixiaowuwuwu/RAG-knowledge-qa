from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.loaders import LoadedDocument


def test_chunk_documents_preserves_metadata():
    document = LoadedDocument(
        text="第一段内容。\n\n第二段内容。",
        source="data/documents/example.md",
        metadata={"file_type": ".md"},
    )

    chunks = chunk_documents([document], chunk_size=20, chunk_overlap=4)

    assert chunks
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].source == "data/documents/example.md"
    assert chunks[0].metadata["file_type"] == ".md"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].content.strip()


def test_chunk_documents_skips_empty_text():
    document = LoadedDocument(text="   ", source="empty.md", metadata={"file_type": ".md"})

    chunks = chunk_documents([document], chunk_size=20, chunk_overlap=4)

    assert chunks == []


def test_chunk_indexes_are_sequential_per_document():
    document = LoadedDocument(
        text="abcdefg hijklmn opqrstu vwxyz " * 5,
        source="alphabet.txt",
        metadata={"file_type": ".txt"},
    )

    chunks = chunk_documents([document], chunk_size=30, chunk_overlap=5)

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
