from app.rag.documents import RetrievedDocument


def _positive_pairs(
    retrieved: list[list[RetrievedDocument]],
    expected_sources: list[list[str]],
):
    for documents, expected in zip(retrieved, expected_sources, strict=False):
        if expected:
            yield documents, expected


def hit_rate_at_k(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    pairs = list(_positive_pairs(retrieved, expected_sources))
    if not pairs:
        return 0.0

    hits = 0
    for documents, expected in pairs:
        expected_set = set(expected)
        retrieved_sources = {document.source for document in documents}
        if retrieved_sources & expected_set:
            hits += 1
    return hits / len(pairs)


def mean_reciprocal_rank(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    pairs = list(_positive_pairs(retrieved, expected_sources))
    if not pairs:
        return 0.0

    total = 0.0
    for documents, expected in pairs:
        expected_set = set(expected)
        reciprocal = 0.0
        for rank, document in enumerate(documents, start=1):
            if document.source in expected_set:
                reciprocal = 1.0 / rank
                break
        total += reciprocal
    return total / len(pairs)


def source_recall(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    expected_total = sum(len(set(sources)) for sources in expected_sources)
    if expected_total == 0:
        return 0.0

    found = 0
    for documents, expected in zip(retrieved, expected_sources, strict=False):
        retrieved_sources = {document.source for document in documents}
        found += len(retrieved_sources & set(expected))
    return found / expected_total


def precision_at_k(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    pairs = list(_positive_pairs(retrieved, expected_sources))
    if not pairs:
        return 0.0

    total = 0.0
    for documents, expected in pairs:
        if not documents:
            continue
        expected_set = set(expected)
        retrieved_sources = _unique_sources(documents)
        hits = sum(1 for source in retrieved_sources if source in expected_set)
        total += hits / len(retrieved_sources)
    return total / len(pairs)


def ndcg_at_k(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    pairs = list(_positive_pairs(retrieved, expected_sources))
    if not pairs:
        return 0.0

    total = 0.0
    for documents, expected in pairs:
        expected_set = set(expected)
        seen_sources: set[str] = set()
        dcg = 0.0
        for rank, source in enumerate(_unique_sources(documents), start=1):
            if source in expected_set and source not in seen_sources:
                dcg += 1.0 / _log2(rank + 1)
                seen_sources.add(source)

        ideal_hits = min(len(expected_set), len(_unique_sources(documents)))
        ideal_dcg = sum(1.0 / _log2(rank + 1) for rank in range(1, ideal_hits + 1))
        if ideal_dcg:
            total += dcg / ideal_dcg
    return total / len(pairs)


def _unique_sources(documents: list[RetrievedDocument]) -> list[str]:
    sources = []
    seen: set[str] = set()
    for document in documents:
        if document.source in seen:
            continue
        seen.add(document.source)
        sources.append(document.source)
    return sources


def negative_rejection_rate(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    negative_rows = [
        documents
        for documents, expected in zip(retrieved, expected_sources, strict=False)
        if not expected
    ]
    if not negative_rows:
        return 0.0
    rejected = sum(1 for documents in negative_rows if not documents)
    return rejected / len(negative_rows)


def _log2(value: int) -> float:
    import math

    return math.log2(value)
