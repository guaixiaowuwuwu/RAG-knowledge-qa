import json
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


def test_ingest_directory_applies_manifest_acl_to_child_bm25_and_parent_corpora(tmp_path: Path):
    documents_dir = tmp_path / "docs"
    documents_dir.mkdir()
    (documents_dir / "sales.md").write_text("销售部门折扣审批流程。", encoding="utf-8")
    (documents_dir / "public.md").write_text("公开知识库说明。", encoding="utf-8")
    manifest_path = tmp_path / "documents_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "pattern": "sales.md",
                        "tenant_id": "tenant-a",
                        "doc_id": "sales-doc",
                        "document_version": "sales-v2",
                        "allowed_department_ids": ["sales"],
                        "is_public": False,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    bm25_path = tmp_path / "bm25.jsonl"
    parent_path = tmp_path / "parents.jsonl"
    vector_store = FakeVectorStore()

    result = ingest_directory(
        documents_dir=documents_dir,
        vector_store=vector_store,
        chunk_size=80,
        chunk_overlap=8,
        reset=True,
        bm25_corpus_path=bm25_path,
        parent_corpus_path=parent_path,
        parent_chunk_size=160,
        parent_chunk_overlap=8,
        manifest_path=manifest_path,
        default_tenant_id="default",
        document_index_version="index-v1",
    )

    assert result.indexed_chunks == len(vector_store.chunks)
    sales_chunk = next(chunk for chunk in vector_store.chunks if chunk.source.endswith("sales.md"))
    public_chunk = next(chunk for chunk in vector_store.chunks if chunk.source.endswith("public.md"))
    assert sales_chunk.metadata["tenant_id"] == "tenant-a"
    assert sales_chunk.metadata["doc_id"] == "sales-doc"
    assert sales_chunk.metadata["document_version"] == "sales-v2"
    assert sales_chunk.metadata["allowed_department_ids"] == '["sales"]'
    assert sales_chunk.metadata["is_public"] is False
    assert public_chunk.metadata["tenant_id"] == "default"
    assert public_chunk.metadata["document_version"] == "index-v1"
    assert public_chunk.metadata["is_public"] is True

    bm25_rows = [json.loads(line) for line in bm25_path.read_text(encoding="utf-8").splitlines()]
    parent_rows = [json.loads(line) for line in parent_path.read_text(encoding="utf-8").splitlines()]
    sales_bm25 = next(row for row in bm25_rows if row["source"].endswith("sales.md"))
    sales_parent = next(row for row in parent_rows if row["source"].endswith("sales.md"))
    assert sales_bm25["metadata"]["allowed_department_ids"] == '["sales"]'
    assert sales_parent["metadata"]["allowed_department_ids"] == '["sales"]'
    assert sales_bm25["metadata"]["tenant_id"] == "tenant-a"
    assert sales_parent["metadata"]["tenant_id"] == "tenant-a"
