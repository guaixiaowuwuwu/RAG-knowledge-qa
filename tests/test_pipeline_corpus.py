import json
from pathlib import Path

from app.ingestion.chunker import Chunk
from app.ingestion.pipeline import persist_bm25_corpus


def test_persist_bm25_corpus_writes_jsonl(tmp_path: Path):
    corpus_path = tmp_path / "bm25.jsonl"
    chunks = [
        Chunk(
            content="RAG 系统包含检索和生成。",
            source="example.md",
            metadata={"source": "example.md", "chunk_index": 0, "file_type": ".md"},
        )
    ]

    persist_bm25_corpus(chunks, corpus_path)

    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "id": rows[0]["id"],
            "content": "RAG 系统包含检索和生成。",
            "source": "example.md",
            "metadata": {"source": "example.md", "chunk_index": 0, "file_type": ".md", "chunk_id": rows[0]["id"]},
        }
    ]
    assert rows[0]["id"]
