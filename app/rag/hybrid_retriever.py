from app.rag.bm25 import BM25Retriever
from app.rag.documents import (
    RetrievedDocument,
    chunk_to_retrieved_document,
    retrieved_document_to_chunk,
)
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.reranker import Reranker


class HybridRetriever:
    def __init__(
        self,
        dense_retriever,
        sparse_retriever: BM25Retriever,
        reranker: Reranker,
        dense_top_k: int,
        sparse_top_k: int,
        rrf_k: int,
        reranker_top_n: int,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.reranker_top_n = reranker_top_n

    def similarity_search(self, query: str, top_k: int):
        dense_chunks = self.dense_retriever.similarity_search(query, top_k=self.dense_top_k)
        dense_documents = [chunk_to_retrieved_document(chunk) for chunk in dense_chunks]
        sparse_documents: list[RetrievedDocument] = self.sparse_retriever.search(query, top_k=self.sparse_top_k)

        fused = reciprocal_rank_fusion(
            [dense_documents, sparse_documents],
            top_k=max(top_k, self.reranker_top_n),
            k=self.rrf_k,
        )
        reranked = self.reranker.rerank(query, fused, top_n=max(top_k, self.reranker_top_n))
        return [retrieved_document_to_chunk(document) for document in reranked[:top_k]]
