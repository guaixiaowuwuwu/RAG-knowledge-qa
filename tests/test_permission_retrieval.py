import json

from app.ingestion.chunker import Chunk
from app.rag.bm25 import BM25Retriever
from app.rag.documents import RetrievedDocument
from app.rag.service import RagService
from app.rag.hybrid_retriever import HybridRetriever, RetrievalDebugOptions
from app.security.acl import RetrievalAccessFilter
from app.security.context import RequestContext


def acl_metadata(
    *,
    tenant_id: str = "tenant-a",
    users=(),
    departments=(),
    roles=(),
    public: bool = False,
) -> dict:
    return {
        "tenant_id": tenant_id,
        "allowed_user_ids": json.dumps(list(users), ensure_ascii=False),
        "allowed_department_ids": json.dumps(list(departments), ensure_ascii=False),
        "allowed_roles": json.dumps(list(roles), ensure_ascii=False),
        "is_public": public,
    }


class ListDenseRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_top_k = None

    def similarity_search(self, query: str, top_k: int, access_filter=None):
        self.last_top_k = top_k
        return self.chunks[:top_k]


class ListSparseRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.last_top_k = None

    def search(self, query: str, top_k: int, access_filter=None):
        self.last_top_k = top_k
        results = self.documents[:top_k]
        if access_filter is not None:
            results = [document for document in results if access_filter.can_access_metadata(document.metadata)]
        return results


class IdentityReranker:
    def rerank(self, query, documents, top_n):
        return [
            RetrievedDocument(
                id=document.id,
                content=document.content,
                source=document.source,
                metadata=dict(document.metadata),
                score=1.0,
            )
            for document in documents[:top_n]
        ]


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return "allowed answer"


def context_for(*, department: str = "sales", tenant_id: str = "tenant-a", user_id: str = "user-1"):
    return RequestContext(
        tenant_id=tenant_id,
        user_id=user_id,
        display_name="User",
        department_ids=(department,),
        roles=("employee",),
        permission_version="perm-v1",
        source="wecom",
    )


def build_retriever(*, dense_chunks=None, sparse_documents=None, parent_store=None):
    return HybridRetriever(
        dense_retriever=ListDenseRetriever(dense_chunks or []),
        sparse_retriever=ListSparseRetriever(sparse_documents or []),
        reranker=IdentityReranker(),
        dense_top_k=2,
        sparse_top_k=2,
        rrf_k=60,
        reranker_top_n=2,
        parent_store=parent_store,
        permission_filter_overfetch_max=6,
    )


def test_department_contexts_return_different_allowed_sources():
    sales_chunk = Chunk(
        content="销售折扣审批流程",
        source="sales.md",
        metadata={"source": "sales.md", "chunk_index": 0, **acl_metadata(departments=("sales",))},
    )
    hr_chunk = Chunk(
        content="人事入职流程",
        source="hr.md",
        metadata={"source": "hr.md", "chunk_index": 0, **acl_metadata(departments=("hr",))},
    )
    retriever = build_retriever(dense_chunks=[sales_chunk, hr_chunk])

    sales_chunks = retriever.similarity_search("流程", top_k=2, context=context_for(department="sales"))
    hr_chunks = retriever.similarity_search("流程", top_k=2, context=context_for(department="hr"))

    assert [chunk.source for chunk in sales_chunks] == ["sales.md"]
    assert [chunk.source for chunk in hr_chunks] == ["hr.md"]
    assert retriever.dense_retriever.last_top_k == 6


def test_cross_tenant_documents_do_not_appear_in_debug_or_sse_sources():
    denied = Chunk(
        content="tenant-b secret payroll",
        source="secret.md",
        metadata={"source": "secret.md", "chunk_index": 0, **acl_metadata(tenant_id="tenant-b", public=True)},
    )
    allowed = Chunk(
        content="tenant-a handbook",
        source="handbook.md",
        metadata={"source": "handbook.md", "chunk_index": 0, **acl_metadata(public=True)},
    )
    retriever = build_retriever(dense_chunks=[denied, allowed])
    result = retriever.similarity_search_with_trace(
        "handbook",
        top_k=1,
        options=RetrievalDebugOptions(reranker_enabled=False),
        context=context_for(),
    )
    trace_json = json.dumps(result.trace.to_dict(), ensure_ascii=False)

    assert [chunk.source for chunk in result.chunks] == ["handbook.md"]
    assert "tenant-b secret payroll" not in trace_json
    assert "secret.md" not in trace_json

    service = RagService(retriever=retriever, llm=FakeLLM())
    events = list(service.answer_stream("handbook", top_k=1, debug=True, context=context_for()))
    sources_event = next(event for event in events if event["event"] == "sources")
    debug_event = next(event for event in events if event["event"] == "debug")
    assert "tenant-b secret payroll" not in sources_event["data"]
    assert "tenant-b secret payroll" not in debug_event["data"]
    assert "handbook.md" in sources_event["data"]


def test_bm25_only_match_respects_acl(tmp_path):
    corpus = tmp_path / "bm25.jsonl"
    rows = [
        {
            "id": "denied",
            "content": "关键词检索 销售机密",
            "source": "denied.md",
            "metadata": {"source": "denied.md", "chunk_id": "denied", **acl_metadata(departments=("finance",))},
        },
        {
            "id": "allowed",
            "content": "关键词检索 销售流程",
            "source": "allowed.md",
            "metadata": {"source": "allowed.md", "chunk_id": "allowed", **acl_metadata(departments=("sales",))},
        },
    ]
    corpus.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    retriever = BM25Retriever.from_jsonl(corpus)
    access_filter = RetrievalAccessFilter.from_context(context_for(department="sales"))

    results = retriever.search("关键词检索", top_k=2, access_filter=access_filter)

    assert [document.source for document in results] == ["allowed.md"]


class DeniedParentStore:
    def get(self, parent_id: str):
        return Chunk(
            content="denied parent secret content",
            source="parent-secret.md",
            metadata={
                "source": "parent-secret.md",
                "chunk_index": 0,
                "parent_id": parent_id,
                **acl_metadata(tenant_id="tenant-b", public=True),
            },
        )


def test_parent_hydration_does_not_leak_denied_parent_content():
    child = Chunk(
        content="allowed child content",
        source="child.md",
        metadata={"source": "child.md", "chunk_index": 0, "parent_id": "parent-1", **acl_metadata(public=True)},
    )
    retriever = build_retriever(dense_chunks=[child], parent_store=DeniedParentStore())

    result = retriever.similarity_search_with_trace(
        "child",
        top_k=1,
        options=RetrievalDebugOptions(reranker_enabled=False),
        context=context_for(),
    )
    trace_json = json.dumps(result.trace.to_dict(), ensure_ascii=False)

    assert result.chunks[0].content == "allowed child content"
    assert "denied parent secret content" not in trace_json
    assert any(row["status"] == "parent_denied" for row in result.trace.parent_hydration)
