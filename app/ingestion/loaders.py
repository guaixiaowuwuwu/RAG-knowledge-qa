from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


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
