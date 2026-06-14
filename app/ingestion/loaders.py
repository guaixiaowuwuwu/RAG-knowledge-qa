from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}


@dataclass(frozen=True)
class LoadedDocument:
    text: str
    source: str
    metadata: dict


@dataclass(frozen=True)
class LoadResult:
    documents: list[LoadedDocument]
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def load_document(path: Path) -> list[LoadedDocument]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix}")

    if suffix in {".txt", ".md"}:
        return [
            LoadedDocument(
                text=path.read_text(encoding="utf-8"),
                source=str(path),
                metadata={"file_type": suffix},
            )
        ]

    if suffix == ".docx":
        return _load_docx(path)

    if suffix in {".html", ".htm"}:
        return _load_html(path)

    return _load_pdf(path)


def load_documents_from_dir(directory: Path) -> LoadResult:
    documents: list[LoadedDocument] = []
    skipped: list[str] = []
    errors: dict[str, str] = {}

    if not directory.exists():
        return LoadResult(documents=[], skipped=[], errors={str(directory): "Directory does not exist"})

    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append(str(path))
            continue

        try:
            documents.extend(load_document(path))
        except Exception as exc:
            errors[str(path)] = str(exc)

    return LoadResult(documents=documents, skipped=skipped, errors=errors)


def _load_pdf(path: Path) -> list[LoadedDocument]:
    reader = PdfReader(str(path))
    documents: list[LoadedDocument] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        documents.append(
            LoadedDocument(
                text=text,
                source=str(path),
                metadata={"file_type": ".pdf", "page": page_index},
            )
        )

    return documents


def _load_docx(path: Path) -> list[LoadedDocument]:
    doc = DocxDocument(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text:
        return []
    return [
        LoadedDocument(
            text=text,
            source=str(path),
            metadata={"file_type": ".docx"},
        )
    ]


def _load_html(path: Path) -> list[LoadedDocument]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = "\n".join(lines)
    if not normalized:
        return []
    return [
        LoadedDocument(
            text=normalized,
            source=str(path),
            metadata={"file_type": path.suffix.lower()},
        )
    ]
