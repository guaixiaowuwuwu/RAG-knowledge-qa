from app.rag.documents import RetrievedDocument


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDocument]],
    top_k: int,
    k: int = 60,
) -> list[RetrievedDocument]:
    scores: dict[str, float] = {}
    documents: dict[str, RetrievedDocument] = {}

    for ranked in ranked_lists:
        for rank, document in enumerate(ranked, start=1):
            documents.setdefault(document.id, document)
            scores[document.id] = scores.get(document.id, 0.0) + 1.0 / (k + rank)

    ordered_ids = sorted(scores, key=lambda identity: scores[identity], reverse=True)
    fused: list[RetrievedDocument] = []
    for identity in ordered_ids[:top_k]:
        document = documents[identity]
        fused.append(
            RetrievedDocument(
                id=document.id,
                content=document.content,
                source=document.source,
                metadata=dict(document.metadata),
                score=scores[identity],
            )
        )
    return fused
