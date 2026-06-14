from app.rag.documents import RetrievedDocument


def hit_rate_at_k(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    if not expected_sources:
        return 0.0

    hits = 0
    for documents, expected in zip(retrieved, expected_sources, strict=False):
        expected_set = set(expected)
        retrieved_sources = {document.source for document in documents}
        if retrieved_sources & expected_set:
            hits += 1
    return hits / len(expected_sources)
