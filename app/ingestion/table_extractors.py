from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from app.ingestion.cleaning import normalize_document_text


@dataclass(frozen=True)
class TableBlock:
    markdown: str
    metadata: dict


def rows_to_markdown(rows: list[list[str]]) -> str:
    normalized_rows = _rectangular_rows(rows)
    if not normalized_rows:
        return ""

    header = normalized_rows[0]
    body = normalized_rows[1:]
    lines = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def extract_html_tables(container: BeautifulSoup | Tag, base_metadata: dict | None = None) -> list[TableBlock]:
    tables: list[TableBlock] = []
    metadata = dict(base_metadata or {})
    for table_index, table in enumerate(container.find_all("table")):
        rows = _rows_from_html_table(table)
        markdown = rows_to_markdown(rows)
        if not markdown:
            continue
        table_metadata = dict(metadata)
        table_metadata.update({"content_type": "table", "table_index": table_index})
        tables.append(TableBlock(markdown=markdown, metadata=table_metadata))
    return tables


def extract_pdf_tables(path: Path) -> list[TableBlock]:
    try:
        import pdfplumber
    except ImportError:
        return []

    tables: list[TableBlock] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            try:
                page_tables = page.extract_tables() or []
            except Exception:
                continue
            for table_index, table in enumerate(page_tables):
                markdown = rows_to_markdown(_clean_table_rows(table))
                if not markdown:
                    continue
                tables.append(
                    TableBlock(
                        markdown=markdown,
                        metadata={"page": page_index, "table_index": table_index, "content_type": "table"},
                    )
                )
    return tables


def docx_table_to_markdown(table: Any) -> str:
    rows = [[normalize_document_text(cell.text) for cell in row.cells] for row in table.rows]
    return rows_to_markdown(rows)


def _rows_from_html_table(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        rows.append([normalize_document_text(cell.get_text(separator=" ")) for cell in cells])
    return rows


def _clean_table_rows(rows: list[list[Any]]) -> list[list[str]]:
    return [[normalize_document_text(str(cell or "")) for cell in row] for row in rows]


def _rectangular_rows(rows: list[list[str]]) -> list[list[str]]:
    cleaned = [[normalize_document_text(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return []

    width = max(len(row) for row in cleaned)
    if width == 0:
        return []
    rectangular = [row + [""] * (width - len(row)) for row in cleaned]
    if len(rectangular) == 1:
        rectangular.append([""] * width)
    return rectangular


def _escape_markdown_cell(cell: str) -> str:
    return normalize_document_text(cell).replace("\n", " ").replace("|", "\\|")
