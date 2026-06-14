from typing import Protocol

from app.rag.documents import RetrievedDocument


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[RetrievedDocument], top_n: int) -> list[RetrievedDocument]:
        ...


class NoopReranker:
    def rerank(self, query: str, documents: list[RetrievedDocument], top_n: int) -> list[RetrievedDocument]:
        return documents[:top_n]


class ScoreBasedReranker:
    def __init__(self, model):
        self.model = model

    def rerank(self, query: str, documents: list[RetrievedDocument], top_n: int) -> list[RetrievedDocument]:
        if not documents:
            return []

        pairs = [[query, document.content] for document in documents]
        scores = self.model.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]

        scored = []
        for document, score in zip(documents, scores, strict=False):
            scored.append(
                RetrievedDocument(
                    id=document.id,
                    content=document.content,
                    source=document.source,
                    metadata=dict(document.metadata),
                    score=float(score),
                )
            )
        scored.sort(key=lambda document: document.score or 0.0, reverse=True)
        return scored[:top_n]


def build_reranker(enabled: bool, model_name: str):
    if not enabled:
        return NoopReranker()

    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError(
            "RERANKER_ENABLED=true requires FlagEmbedding. "
            "Install it separately with `pip install FlagEmbedding`."
        ) from exc

    return ScoreBasedReranker(FlagReranker(model_name, use_fp16=True))
