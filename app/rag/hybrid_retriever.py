import inspect
import time
from dataclasses import dataclass, field

from app.ingestion.chunker import Chunk
from app.observability.metrics import metrics
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
from app.security.acl import RetrievalAccessFilter
from app.security.context import RequestContext


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
        permission_filter_overfetch_max: int = 100,
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
        self.permission_filter_overfetch_max = permission_filter_overfetch_max

    def similarity_search(
        self,
        query: str,
        top_k: int,
        *,
        access_filter: RetrievalAccessFilter | None = None,
        context: RequestContext | None = None,
    ):
        return self.similarity_search_with_trace(
            query,
            top_k=top_k,
            access_filter=access_filter,
            context=context,
        ).chunks

    def similarity_search_with_trace(
        self,
        query: str,
        top_k: int,
        *,
        options: RetrievalDebugOptions | None = None,
        access_filter: RetrievalAccessFilter | None = None,
        context: RequestContext | None = None,
    ) -> RetrievalResult:
        options = options or RetrievalDebugOptions()
        if access_filter is None and context is not None:
            access_filter = RetrievalAccessFilter.from_context(context)
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
            dense_query_top_k = self._overfetch_top_k(self.dense_top_k, access_filter)
            dense_chunks = self._dense_search(
                expanded_query,
                top_k=dense_query_top_k,
                access_filter=access_filter,
            )
            dense_documents = [
                chunk_to_retrieved_document(chunk)
                for chunk in self._filter_chunks(dense_chunks, access_filter)[: self.dense_top_k]
            ]
            sparse_query_top_k = self._overfetch_top_k(self.sparse_top_k, access_filter)
            sparse_documents: list[RetrievedDocument] = self.sparse_retriever.search(
                expanded_query,
                top_k=sparse_query_top_k,
                **self._access_filter_kwargs(self.sparse_retriever.search, access_filter),
            )
            sparse_documents = self._filter_documents(sparse_documents, access_filter)[: self.sparse_top_k]
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
            fused_chunks, parent_debug = self._hydrate_with_debug(fused_chunks, access_filter=access_filter)
        fused_chunks = self._filter_chunks(fused_chunks, access_filter)
        fused = [chunk_to_retrieved_document(chunk) for chunk in fused_chunks]
        reranker_scores: list[dict] = []
        if options.reranker_enabled:
            reranker_started_at = time.perf_counter()
            reranked = self.reranker.rerank(query, fused, top_n=max(top_k, self.reranker_top_n))
            metrics.observe("rag_reranker_latency_ms", round((time.perf_counter() - reranker_started_at) * 1000, 3))
            reranked = self._filter_documents(reranked, access_filter)
            reranker_scores = self._documents_debug(reranked, stage="reranker")
        else:
            reranked = self._filter_documents(fused, access_filter)[: max(top_k, self.reranker_top_n)]
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
            final_chunks=self._documents_debug(final_documents, stage="final"),
            options={
                "rewrite_enabled": options.rewrite_enabled,
                "hyde_enabled": options.hyde_enabled,
                "parent_hydration_enabled": parent_hydration_enabled,
                "reranker_enabled": options.reranker_enabled,
                "max_query_variants": options.max_query_variants,
                "access_filter": access_filter.summary() if access_filter is not None else None,
            },
        )
        return RetrievalResult(chunks=final_chunks, trace=trace)

    def _dense_search(
        self,
        query: str,
        *,
        top_k: int,
        access_filter: RetrievalAccessFilter | None,
    ) -> list[Chunk]:
        return self.dense_retriever.similarity_search(
            query,
            top_k=top_k,
            **self._access_filter_kwargs(self.dense_retriever.similarity_search, access_filter),
        )

    def _access_filter_kwargs(self, method, access_filter: RetrievalAccessFilter | None) -> dict:
        if access_filter is None:
            return {}
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return {}
        if "access_filter" in parameters:
            return {"access_filter": access_filter}
        return {}

    def _filter_chunks(
        self,
        chunks: list[Chunk],
        access_filter: RetrievalAccessFilter | None,
    ) -> list[Chunk]:
        if access_filter is None:
            return chunks
        return [chunk for chunk in chunks if access_filter.can_access_metadata(chunk.metadata)]

    def _filter_documents(
        self,
        documents: list[RetrievedDocument],
        access_filter: RetrievalAccessFilter | None,
    ) -> list[RetrievedDocument]:
        if access_filter is None:
            return documents
        return [document for document in documents if access_filter.can_access_metadata(document.metadata)]

    def _overfetch_top_k(self, top_k: int, access_filter: RetrievalAccessFilter | None) -> int:
        if access_filter is None or top_k <= 0:
            return top_k
        return min(max(top_k * 3, top_k), self.permission_filter_overfetch_max)

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
                    "table_index": document.metadata.get("table_index"),
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
            "table_index": chunk.metadata.get("table_index"),
            "content": chunk.content,
        }

    def _hydrate_with_debug(
        self,
        children,
        access_filter: RetrievalAccessFilter | None = None,
    ) -> tuple[list, list[dict]]:
        if not hasattr(self.parent_store, "get") and hasattr(self.parent_store, "hydrate"):
            hydrate_kwargs = self._access_filter_kwargs(self.parent_store.hydrate, access_filter)
            hydrated = self.parent_store.hydrate(children, **hydrate_kwargs)
            hydrated = self._filter_chunks(hydrated, access_filter)
            return hydrated, [
                {
                    "status": "hydrated_by_parent_store",
                    "child_count": len(children),
                    "hydrated_count": len(hydrated),
                }
            ]

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
            if access_filter is not None and not access_filter.can_access_metadata(parent.metadata):
                hydrated.append(child)
                trace.append(
                    {
                        "status": "parent_denied",
                        "child": child_row,
                        "parent": {
                            "source": parent.source,
                            "page": parent.metadata.get("page"),
                            "chunk_index": parent.metadata.get("chunk_index"),
                            "parent_id": parent_id,
                        },
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
