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


def page_hit_rate_at_k(
    retrieved: list[list[RetrievedDocument]],
    expected_sources: list[list[str]],
    expected_pages: list[dict[str, list[int]]],
) -> float:
    rows = [
        (documents, expected, pages)
        for documents, expected, pages in zip(retrieved, expected_sources, expected_pages, strict=False)
        if pages
    ]
    if not rows:
        return 0.0

    hits = sum(1 for documents, expected, pages in rows if case_page_hit(documents, expected, pages))
    return hits / len(rows)


def evidence_keyword_recall_at_k(
    retrieved: list[list[RetrievedDocument]],
    expected_chunk_keywords: list[list[str]],
) -> float:
    rows = [
        (documents, keywords)
        for documents, keywords in zip(retrieved, expected_chunk_keywords, strict=False)
        if keywords
    ]
    if not rows:
        return 0.0

    return sum(case_evidence_keyword_recall(documents, keywords) for documents, keywords in rows) / len(rows)


def evidence_strict_hit_at_k(
    retrieved: list[list[RetrievedDocument]],
    expected_sources: list[list[str]],
    expected_pages: list[dict[str, list[int]]],
    expected_chunk_keywords: list[list[str]],
) -> float:
    rows = [
        (documents, sources, pages, keywords)
        for documents, sources, pages, keywords in zip(
            retrieved,
            expected_sources,
            expected_pages,
            expected_chunk_keywords,
            strict=False,
        )
        if sources and (pages or keywords)
    ]
    if not rows:
        return 0.0

    hits = sum(
        1
        for documents, sources, pages, keywords in rows
        if case_evidence_strict_hit(documents, sources, pages, keywords)
    )
    return hits / len(rows)


def case_page_hit(
    documents: list[RetrievedDocument],
    expected_sources: list[str],
    expected_pages: dict[str, list[int]],
) -> bool | None:
    if not expected_pages:
        return None

    expected_source_set = set(expected_sources)
    for document in documents:
        if document.source not in expected_source_set:
            continue
        expected_for_source = expected_pages.get(document.source)
        if not expected_for_source:
            continue
        page = document.metadata.get("page")
        if page is not None and int(page) in expected_for_source:
            return True
    return False


def case_evidence_keyword_matches(
    documents: list[RetrievedDocument],
    expected_chunk_keywords: list[str],
) -> tuple[list[str], list[str]]:
    if not expected_chunk_keywords:
        return [], []

    haystack = "\n".join(document.content for document in documents).lower()
    matches = [keyword for keyword in expected_chunk_keywords if keyword.lower() in haystack]
    misses = [keyword for keyword in expected_chunk_keywords if keyword.lower() not in haystack]
    return matches, misses


def case_evidence_keyword_recall(
    documents: list[RetrievedDocument],
    expected_chunk_keywords: list[str],
) -> float:
    if not expected_chunk_keywords:
        return 0.0
    matches, _misses = case_evidence_keyword_matches(documents, expected_chunk_keywords)
    return len(matches) / len(expected_chunk_keywords)


def case_evidence_strict_hit(
    documents: list[RetrievedDocument],
    expected_sources: list[str],
    expected_pages: dict[str, list[int]],
    expected_chunk_keywords: list[str],
) -> bool | None:
    if not expected_sources or not (expected_pages or expected_chunk_keywords):
        return None

    retrieved_sources = {document.source for document in documents}
    source_hit = bool(retrieved_sources & set(expected_sources))
    page_hit = case_page_hit(documents, expected_sources, expected_pages) if expected_pages else True
    _matches, misses = case_evidence_keyword_matches(documents, expected_chunk_keywords)
    keyword_hit = not misses if expected_chunk_keywords else True
    return source_hit and bool(page_hit) and keyword_hit


def refusal_reason_counts(refusal_reasons: list[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in refusal_reasons:
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _log2(value: int) -> float:
    import math

    return math.log2(value)
