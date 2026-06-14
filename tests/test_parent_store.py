from pathlib import Path

from app.ingestion.chunker import Chunk, ParentChildChunks, chunk_documents_with_parents
from app.ingestion.loaders import LoadedDocument
from app.rag.parent_store import JsonlParentStore


def test_chunk_documents_with_parents_links_children_to_parent():
    document = LoadedDocument(
        text="第一段介绍 RAG 系统。\n\n第二段介绍混合检索。\n\n第三段介绍 reranker。",
        source="example.md",
        metadata={"file_type": ".md"},
    )

    result = chunk_documents_with_parents(
        [document],
        child_chunk_size=24,
        child_chunk_overlap=4,
        parent_chunk_size=80,
        parent_chunk_overlap=8,
    )

    assert isinstance(result, ParentChildChunks)
    assert result.parents
    assert result.children
    assert all("parent_id" in child.metadata for child in result.children)
    assert {child.metadata["parent_id"] for child in result.children}.issubset(
        {parent.metadata["parent_id"] for parent in result.parents}
    )


def test_jsonl_parent_store_round_trips_parent_chunks(tmp_path: Path):
    path = tmp_path / "parents.jsonl"
    parent = Chunk(
        content="父块上下文内容",
        source="example.md",
        metadata={"source": "example.md", "parent_id": "parent-1", "chunk_index": 0},
    )

    store = JsonlParentStore(path)
    store.write([parent])
    loaded = JsonlParentStore(path)

    result = loaded.get("parent-1")

    assert result is not None
    assert result.content == "父块上下文内容"
    assert result.metadata["parent_id"] == "parent-1"
