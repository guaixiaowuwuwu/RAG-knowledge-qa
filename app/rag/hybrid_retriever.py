from dataclasses import dataclass, field

from app.rag.bm25 import BM25Retriever
from app.rag.documents import (
    RetrievedDocument,
    chunk_to_retrieved_document,
    chunk_id,
    retrieved_document_to_chunk,
)
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.parent_store import JsonlParentStore
from app.rag.query_transform import QueryTransformer
from app.rag.reranker import Reranker


@dataclass(frozen=True)
class RetrievalDebugOptions:
    rewrite_enabled: bool | None = None
    hyde_enabled: bool | None = None
    parent_hydration_enabled: bool | None = None
    reranker_enabled: bool = True
    max_query_variants: int | None = None


@dataclass(frozen=True)
class RetrievalTrace:
    query: str
    query_variants: list[str]
    dense_candidates: list[dict] = field(default_factory=list)
    bm25_candidates: list[dict] = field(default_factory=list)
    rrf_scores: list[dict] = field(default_factory=list)
    reranker_scores: list[dict] = field(default_factory=list)
    parent_hydration: list[dict] = field(default_factory=list)
    final_chunks: list[dict] = field(default_factory=list)
    options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_variants": self.query_variants,
            "dense_candidates": self.dense_candidates,
            "bm25_candidates": self.bm25_candidates,
            "rrf_scores": self.rrf_scores,
            "reranker_scores": self.reranker_scores,
            "parent_hydration": self.parent_hydration,
            "final_chunks": self.final_chunks,
            "options": self.options,
        }


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list
    trace: RetrievalTrace


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
        return self.similarity_search_with_trace(query, top_k=top_k).chunks

    def similarity_search_with_trace(
        self,
        query: str,
        top_k: int,
        *,
        options: RetrievalDebugOptions | None = None,
    ) -> RetrievalResult:
        options = options or RetrievalDebugOptions()
        parent_hydration_enabled = self.parent_store is not None
        if options.parent_hydration_enabled is not None:
            parent_hydration_enabled = options.parent_hydration_enabled

        if self.query_transformer is not None:
            queries = self.query_transformer.expand(
                query,
                rewrite_enabled=options.rewrite_enabled,
                hyde_enabled=options.hyde_enabled,
                max_variants=options.max_query_variants,
            )
        else:
            queries = [query]

        ranked_lists = []
        dense_debug: list[dict] = []
        sparse_debug: list[dict] = []
        for expanded_query in queries:
            dense_chunks = self.dense_retriever.similarity_search(expanded_query, top_k=self.dense_top_k)
            dense_documents = [chunk_to_retrieved_document(chunk) for chunk in dense_chunks]
            sparse_documents: list[RetrievedDocument] = self.sparse_retriever.search(
                expanded_query,
                top_k=self.sparse_top_k,
            )
            ranked_lists.extend([dense_documents, sparse_documents])
            dense_debug.extend(
                self._documents_debug(
                    dense_documents,
                    stage="dense",
                    query_variant=expanded_query,
                )
            )
            sparse_debug.extend(
                self._documents_debug(
                    sparse_documents,
                    stage="bm25",
                    query_variant=expanded_query,
                )
            )

        fusion_top_k = max(
            top_k,
            self.reranker_top_n,
            self.dense_top_k,
            self.sparse_top_k,
        )
        fused = reciprocal_rank_fusion(
            ranked_lists,
            top_k=fusion_top_k,
            k=self.rrf_k,
        )
        rrf_debug = self._documents_debug(fused, stage="rrf")
        fused_chunks = [retrieved_document_to_chunk(document) for document in fused]
        parent_debug: list[dict] = []
        if parent_hydration_enabled and self.parent_store is not None:
            fused_chunks, parent_debug = self._hydrate_with_debug(fused_chunks)
        fused = [chunk_to_retrieved_document(chunk) for chunk in fused_chunks]
        reranker_scores: list[dict] = []
        if options.reranker_enabled:
            reranked = self.reranker.rerank(query, fused, top_n=max(top_k, self.reranker_top_n))
            reranker_scores = self._documents_debug(reranked, stage="reranker")
        else:
            reranked = fused[: max(top_k, self.reranker_top_n)]
        final_documents = reranked[:top_k]
        final_chunks = [retrieved_document_to_chunk(document) for document in final_documents]
        trace = RetrievalTrace(
            query=query,
            query_variants=queries,
            dense_candidates=dense_debug,
            bm25_candidates=sparse_debug,
            rrf_scores=rrf_debug,
            reranker_scores=reranker_scores,
            parent_hydration=parent_debug,
            final_chunks=[self._chunk_debug(chunk) for chunk in final_chunks],
            options={
                "rewrite_enabled": options.rewrite_enabled,
                "hyde_enabled": options.hyde_enabled,
                "parent_hydration_enabled": parent_hydration_enabled,
                "reranker_enabled": options.reranker_enabled,
                "max_query_variants": options.max_query_variants,
            },
        )
        return RetrievalResult(chunks=final_chunks, trace=trace)

    def _documents_debug(
        self,
        documents: list[RetrievedDocument],
        *,
        stage: str,
        query_variant: str | None = None,
    ) -> list[dict]:
        rows = []
        for rank, document in enumerate(documents, start=1):
            rows.append(
                {
                    "stage": stage,
                    "rank": rank,
                    "id": document.id,
                    "source": document.source,
                    "score": document.score,
                    "query_variant": query_variant,
                    "page": document.metadata.get("page"),
                    "chunk_index": document.metadata.get("chunk_index"),
                    "matched_child_chunk_index": document.metadata.get("matched_child_chunk_index"),
                    "content_type": document.metadata.get("content_type"),
                    "content": document.content,
                }
            )
        return rows

    def _chunk_debug(self, chunk) -> dict:
        return {
            "id": str(chunk.metadata.get("chunk_id") or chunk_id(chunk)),
            "source": chunk.source,
            "page": chunk.metadata.get("page"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "matched_child_chunk_index": chunk.metadata.get("matched_child_chunk_index"),
            "content_type": chunk.metadata.get("content_type"),
            "content": chunk.content,
        }

    def _hydrate_with_debug(self, children) -> tuple[list, list[dict]]:
        hydrated = []
        trace = []
        seen_parent_ids: set[str] = set()
        for child in children:
            parent_id = child.metadata.get("parent_id")
            parent = self.parent_store.get(str(parent_id)) if parent_id else None
            child_row = self._chunk_debug(child)
            if parent is None:
                hydrated.append(child)
                trace.append(
                    {
                        "status": "child_kept",
                        "child": child_row,
                        "parent_id": parent_id,
                    }
                )
                continue
            if str(parent_id) in seen_parent_ids:
                trace.append(
                    {
                        "status": "duplicate_parent_skipped",
                        "child": child_row,
                        "parent_id": parent_id,
                    }
                )
                continue
            seen_parent_ids.add(str(parent_id))
            metadata = dict(parent.metadata)
            metadata["matched_child_chunk_index"] = child.metadata.get("chunk_index")
            hydrated_parent = type(child)(
                content=parent.content,
                source=parent.source,
                metadata=metadata,
            )
            hydrated.append(hydrated_parent)
            trace.append(
                {
                    "status": "parent_hydrated",
                    "child": child_row,
                    "parent": self._chunk_debug(hydrated_parent),
                    "parent_id": parent_id,
                }
            )
        return hydrated, trace
