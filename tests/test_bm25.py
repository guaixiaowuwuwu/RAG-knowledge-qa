import json
from pathlib import Path

from app.rag.bm25 import BM25Retriever, tokenize


def test_tokenize_handles_chinese_and_ascii():
    tokens = tokenize("RAG 系统支持 BM25 检索")

    assert "rag" in tokens
    assert "bm25" in tokens
    assert "检索" in tokens


def test_bm25_retriever_returns_ranked_documents(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {
            "id": "a",
            "content": "RAG 系统支持向量检索和关键词检索。",
            "source": "a.md",
            "metadata": {"source": "a.md", "chunk_index": 0, "chunk_id": "a"},
        },
        {
            "id": "b",
            "content": "员工报销流程需要提交发票。",
            "source": "b.md",
            "metadata": {"source": "b.md", "chunk_index": 0, "chunk_id": "b"},
        },
    ]
    corpus.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    retriever = BM25Retriever.from_jsonl(corpus)
    results = retriever.search("关键词检索", top_k=1)

    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].source == "a.md"
    assert results[0].score is not None
