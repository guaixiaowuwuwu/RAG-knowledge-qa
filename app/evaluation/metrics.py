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


def mean_reciprocal_rank(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    if not expected_sources:
        return 0.0

    total = 0.0
    for documents, expected in zip(retrieved, expected_sources, strict=False):
        expected_set = set(expected)
        reciprocal = 0.0
        for rank, document in enumerate(documents, start=1):
            if document.source in expected_set:
                reciprocal = 1.0 / rank
                break
        total += reciprocal
    return total / len(expected_sources)


def source_recall(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    expected_total = sum(len(set(sources)) for sources in expected_sources)
    if expected_total == 0:
        return 0.0

    found = 0
    for documents, expected in zip(retrieved, expected_sources, strict=False):
        retrieved_sources = {document.source for document in documents}
        found += len(retrieved_sources & set(expected))
    return found / expected_total
