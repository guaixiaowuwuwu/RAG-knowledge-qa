from app.ingestion.chunker import Chunk
from app.rag.documents import RetrievedDocument
from app.rag.hybrid_retriever import HybridRetriever, RetrievalDebugOptions


class FakeDenseRetriever:
    def similarity_search(self, query: str, top_k: int):
        assert query == "RAG 检索"
        assert top_k == 2
        return [
            Chunk(content="向量检索内容", source="dense.md", metadata={"source": "dense.md", "chunk_index": 0}),
            Chunk(content="重复内容", source="same.md", metadata={"source": "same.md", "chunk_index": 1}),
        ]


class FakeSparseRetriever:
    def search(self, query: str, top_k: int):
        assert query == "RAG 检索"
        assert top_k == 2
        return [
            RetrievedDocument(id="same", content="重复内容", source="same.md", metadata={"chunk_id": "same", "chunk_index": 1}),
            RetrievedDocument(id="sparse", content="关键词检索内容", source="sparse.md", metadata={"chunk_id": "sparse", "chunk_index": 0}),
        ]


class FakeReranker:
    def __init__(self):
        self.called = False
        self.document_count = 0

    def rerank(self, query, documents, top_n):
        self.called = True
        self.document_count = len(documents)
        assert query == "RAG 检索"
        assert top_n == 2
        return list(reversed(documents))[:top_n]


def test_hybrid_retriever_requires_rerank_after_fusion():
    reranker = FakeReranker()
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        sparse_retriever=FakeSparseRetriever(),
        reranker=reranker,
        dense_top_k=2,
        sparse_top_k=2,
        rrf_k=60,
        reranker_top_n=2,
    )

    chunks = retriever.similarity_search("RAG 检索", top_k=2)

    assert reranker.called is True
    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].content in {"向量检索内容", "关键词检索内容", "重复内容"}


class ManyDenseRetriever:
    def similarity_search(self, query: str, top_k: int):
        return [
            Chunk(content=f"dense {index}", source=f"dense-{index}.md", metadata={"source": f"dense-{index}.md", "chunk_index": index})
            for index in range(top_k)
        ]


class EmptySparseRetriever:
    def search(self, query: str, top_k: int):
        return []


class CaptureReranker:
    def __init__(self):
        self.documents = []

    def rerank(self, query, documents, top_n):
        self.documents = documents
        return documents[:top_n]


def test_hybrid_retriever_keeps_wider_fusion_pool_for_reranker():
    reranker = CaptureReranker()
    retriever = HybridRetriever(
        dense_retriever=ManyDenseRetriever(),
        sparse_retriever=EmptySparseRetriever(),
        reranker=reranker,
        dense_top_k=6,
        sparse_top_k=0,
        rrf_k=60,
        reranker_top_n=2,
    )

    chunks = retriever.similarity_search("营业额和同比增幅", top_k=2)

    assert len(reranker.documents) == 6
    assert len(chunks) == 2


class TraceDenseRetriever:
    def similarity_search(self, query: str, top_k: int):
        return [
            Chunk(
                content=f"dense {query}",
                source="dense.md",
                metadata={"source": "dense.md", "chunk_index": 0, "parent_id": "parent-a"},
            )
        ]


class TraceSparseRetriever:
    def search(self, query: str, top_k: int):
        return [
            RetrievedDocument(
                id="sparse-a",
                content=f"sparse {query}",
                source="sparse.md",
                metadata={"chunk_id": "sparse-a", "chunk_index": 1},
                score=3.5,
            )
        ]


class TraceReranker:
    def rerank(self, query, documents, top_n):
        return [
            RetrievedDocument(
                id=document.id,
                content=document.content,
                source=document.source,
                metadata=dict(document.metadata),
                score=1.0 - (index * 0.1),
            )
            for index, document in enumerate(documents[:top_n])
        ]


class TraceQueryTransformer:
    def expand(self, query, **kwargs):
        assert kwargs["rewrite_enabled"] is False
        assert kwargs["hyde_enabled"] is True
        assert kwargs["max_variants"] == 2
        return [query, f"{query} rewritten"]


class TraceParentStore:
    def get(self, parent_id: str):
        if parent_id != "parent-a":
            return None
        return Chunk(
            content="parent dense content",
            source="dense.md",
            metadata={"source": "dense.md", "chunk_index": 99, "parent_id": parent_id},
        )


def test_hybrid_retriever_returns_trace_for_debug_metadata():
    retriever = HybridRetriever(
        dense_retriever=TraceDenseRetriever(),
        sparse_retriever=TraceSparseRetriever(),
        reranker=TraceReranker(),
        dense_top_k=1,
        sparse_top_k=1,
        rrf_k=60,
        reranker_top_n=2,
        parent_store=TraceParentStore(),
        query_transformer=TraceQueryTransformer(),
    )

    result = retriever.similarity_search_with_trace(
        "RAG 检索",
        top_k=2,
        options=RetrievalDebugOptions(
            rewrite_enabled=False,
            hyde_enabled=True,
            max_query_variants=2,
        ),
    )
    trace = result.trace.to_dict()

    assert trace["query_variants"] == ["RAG 检索", "RAG 检索 rewritten"]
    assert trace["dense_candidates"][0]["source"] == "dense.md"
    assert trace["bm25_candidates"][0]["score"] == 3.5
    assert trace["rrf_scores"][0]["score"] is not None
    assert trace["reranker_scores"][0]["score"] is not None
    assert any(item["status"] == "parent_hydrated" for item in trace["parent_hydration"])
    assert trace["final_chunks"][0]["source"] in {"dense.md", "sparse.md"}


def test_hybrid_retriever_can_disable_parent_hydration_and_reranker_for_benchmarks():
    reranker = TraceReranker()
    retriever = HybridRetriever(
        dense_retriever=TraceDenseRetriever(),
        sparse_retriever=TraceSparseRetriever(),
        reranker=reranker,
        dense_top_k=1,
        sparse_top_k=1,
        rrf_k=60,
        reranker_top_n=2,
        parent_store=TraceParentStore(),
    )

    result = retriever.similarity_search_with_trace(
        "RAG 检索",
        top_k=2,
        options=RetrievalDebugOptions(
            parent_hydration_enabled=False,
            reranker_enabled=False,
        ),
    )

    assert result.trace.parent_hydration == []
    assert result.trace.options["parent_hydration_enabled"] is False
    assert result.trace.options["reranker_enabled"] is False
    assert result.chunks[0].content != "parent dense content"
