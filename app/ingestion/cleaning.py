import re
from collections import defaultdict


_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\r\n]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_document_text(text: str) -> str:
    """Normalize text while preserving paragraph and heading boundaries."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [_HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def clean_page_texts(page_texts: list[str], edge_lines: int = 2, min_repetitions: int = 2) -> list[str]:
    """Remove repeated page-edge lines that are likely headers or footers."""
    normalized_pages = [normalize_document_text(page_text) for page_text in page_texts]
    line_pages: defaultdict[str, set[int]] = defaultdict(set)

    for page_index, page_text in enumerate(normalized_pages):
        lines = [line for line in page_text.splitlines() if line.strip()]
        candidates = lines[:edge_lines] + lines[-edge_lines:]
        for candidate in candidates:
            line_pages[candidate].add(page_index)

    repeated = {line for line, page_indexes in line_pages.items() if len(page_indexes) >= min_repetitions}
    cleaned_pages: list[str] = []
    for page_text in normalized_pages:
        lines = page_text.splitlines()
        cleaned_lines: list[str] = []
        last_index = len(lines) - 1
        for index, line in enumerate(lines):
            is_page_edge = index < edge_lines or index > last_index - edge_lines
            if is_page_edge and line in repeated:
                continue
            cleaned_lines.append(line)
        cleaned_pages.append(normalize_document_text("\n".join(cleaned_lines)))

    return cleaned_pages
