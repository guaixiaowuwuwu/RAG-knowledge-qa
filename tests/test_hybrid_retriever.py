from app.ingestion.chunker import Chunk
from app.rag.documents import RetrievedDocument
from app.rag.hybrid_retriever import HybridRetriever


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

    def rerank(self, query, documents, top_n):
        self.called = True
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
