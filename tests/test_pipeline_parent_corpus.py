from pathlib import Path

from app.ingestion.pipeline import ingest_directory


class FakeVectorStore:
    def __init__(self):
        self.reset_called = False
        self.chunks = []

    def reset(self):
        self.reset_called = True

    def add_chunks(self, chunks):
        self.chunks = chunks
        return len(chunks)


def test_ingest_directory_persists_parent_corpus(tmp_path: Path):
    documents = tmp_path / "docs"
    documents.mkdir()
    (documents / "guide.md").write_text(
        "第一段介绍 RAG。\n\n第二段介绍 BM25。\n\n第三段介绍 Reranker。",
        encoding="utf-8",
    )
    parent_path = tmp_path / "parents.jsonl"
    bm25_path = tmp_path / "bm25.jsonl"

    vector_store = FakeVectorStore()
    result = ingest_directory(
        documents_dir=documents,
        vector_store=vector_store,
        chunk_size=24,
        chunk_overlap=4,
        reset=True,
        bm25_corpus_path=bm25_path,
        parent_corpus_path=parent_path,
        parent_chunk_size=80,
        parent_chunk_overlap=8,
    )

    assert result.indexed_chunks > 0
    assert parent_path.exists()
    assert bm25_path.exists()
    assert all("parent_id" in chunk.metadata for chunk in vector_store.chunks)
