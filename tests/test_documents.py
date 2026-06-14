from app.ingestion.chunker import Chunk
from app.rag.documents import RetrievedDocument, chunk_id, chunk_to_retrieved_document


def test_chunk_id_is_stable_for_same_source_and_index():
    chunk = Chunk(
        content="same content",
        source="data/documents/example.md",
        metadata={"chunk_index": 3, "source": "data/documents/example.md"},
    )

    assert chunk_id(chunk) == chunk_id(chunk)


def test_chunk_to_retrieved_document_sets_identity():
    chunk = Chunk(
        content="RAG 知识",
        source="data/documents/example.md",
        metadata={"chunk_index": 1, "source": "data/documents/example.md", "page": 2},
    )

    doc = chunk_to_retrieved_document(chunk, score=0.42)

    assert isinstance(doc, RetrievedDocument)
    assert doc.id == chunk_id(chunk)
    assert doc.content == "RAG 知识"
    assert doc.source == "data/documents/example.md"
    assert doc.metadata["page"] == 2
    assert doc.score == 0.42
