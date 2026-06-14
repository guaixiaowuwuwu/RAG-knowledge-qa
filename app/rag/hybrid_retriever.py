from app.rag.bm25 import BM25Retriever
from app.rag.documents import (
    RetrievedDocument,
    chunk_to_retrieved_document,
    retrieved_document_to_chunk,
)
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.parent_store import JsonlParentStore
from app.rag.query_transform import QueryTransformer
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
        parent_store: JsonlParentStore | None = None,
        query_transformer: QueryTransformer | None = None,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.reranker_top_n = reranker_top_n
        self.parent_store = parent_store
        self.query_transformer = query_transformer

    def similarity_search(self, query: str, top_k: int):
        queries = self.query_transformer.expand(query) if self.query_transformer is not None else [query]

        ranked_lists = []
        for expanded_query in queries:
            dense_chunks = self.dense_retriever.similarity_search(expanded_query, top_k=self.dense_top_k)
            dense_documents = [chunk_to_retrieved_document(chunk) for chunk in dense_chunks]
            sparse_documents: list[RetrievedDocument] = self.sparse_retriever.search(
                expanded_query,
                top_k=self.sparse_top_k,
            )
            ranked_lists.extend([dense_documents, sparse_documents])

        fused = reciprocal_rank_fusion(
            ranked_lists,
            top_k=max(top_k, self.reranker_top_n),
            k=self.rrf_k,
        )
        fused_chunks = [retrieved_document_to_chunk(document) for document in fused]
        if self.parent_store is not None:
            fused_chunks = self.parent_store.hydrate(fused_chunks)
        fused = [chunk_to_retrieved_document(chunk) for chunk in fused_chunks]
        reranked = self.reranker.rerank(query, fused, top_n=max(top_k, self.reranker_top_n))
        return [retrieved_document_to_chunk(document) for document in reranked[:top_k]]
