from app.rag.documents import RetrievedDocument
from app.rag.fusion import reciprocal_rank_fusion


def doc(identity: str, content: str = "content") -> RetrievedDocument:
    return RetrievedDocument(id=identity, content=content, source=f"{identity}.md", metadata={"chunk_id": identity})


def test_rrf_deduplicates_and_prefers_documents_ranked_by_multiple_retrievers():
    dense = [doc("a"), doc("b")]
    sparse = [doc("b"), doc("c")]

    fused = reciprocal_rank_fusion([dense, sparse], top_k=3, k=60)

    assert [item.id for item in fused] == ["b", "a", "c"]
    assert fused[0].score is not None


def test_rrf_handles_empty_lists():
    fused = reciprocal_rank_fusion([[], [doc("x")]], top_k=2, k=60)

    assert [item.id for item in fused] == ["x"]
