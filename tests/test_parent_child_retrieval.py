from app.ingestion.chunker import Chunk
from app.rag.hybrid_retriever import HybridRetriever


class FakeDenseRetriever:
    def similarity_search(self, query: str, top_k: int):
        return [
            Chunk(
                content="子块：BM25",
                source="guide.md",
                metadata={"source": "guide.md", "chunk_index": 0, "parent_id": "parent-1"},
            )
        ]


class FakeSparseRetriever:
    def search(self, query: str, top_k: int):
        return []


class FakeParentStore:
    def hydrate(self, chunks):
        return [
            Chunk(
                content="父块：RAG 系统包含 BM25、RRF 和 Reranker 的完整上下文。",
                source="guide.md",
                metadata={"source": "guide.md", "chunk_index": 0, "parent_id": "parent-1"},
            )
        ]


class FakeReranker:
    def rerank(self, query, documents, top_n):
        return documents[:top_n]


def test_hybrid_retriever_hydrates_parent_context_before_rerank():
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        sparse_retriever=FakeSparseRetriever(),
        reranker=FakeReranker(),
        dense_top_k=1,
        sparse_top_k=1,
        rrf_k=60,
        reranker_top_n=1,
        parent_store=FakeParentStore(),
    )

    chunks = retriever.similarity_search("BM25 是什么？", top_k=1)

    assert chunks[0].content.startswith("父块")
