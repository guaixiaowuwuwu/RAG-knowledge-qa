from pathlib import Path

import pytest

from app.ingestion.loaders import LoadedDocument, load_document, load_documents_from_dir


def test_load_markdown_document(tmp_path: Path):
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nKnowledge text.", encoding="utf-8")

    docs = load_document(path)

    assert docs == [
        LoadedDocument(
            text="# Title\n\nKnowledge text.",
            source=str(path),
            metadata={"file_type": ".md"},
        )
    ]


def test_load_text_document(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("Plain knowledge text.", encoding="utf-8")

    docs = load_document(path)

    assert docs[0].text == "Plain knowledge text."
    assert docs[0].source == str(path)
    assert docs[0].metadata == {"file_type": ".txt"}


def test_unsupported_document_type_raises(tmp_path: Path):
    path = tmp_path / "notes.csv"
    path.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(path)


def test_load_documents_from_dir_skips_unsupported_files(tmp_path: Path):
    supported = tmp_path / "supported.md"
    unsupported = tmp_path / "unsupported.csv"
    supported.write_text("Supported text", encoding="utf-8")
    unsupported.write_text("Unsupported text", encoding="utf-8")

    result = load_documents_from_dir(tmp_path)

    assert len(result.documents) == 1
    assert result.documents[0].text == "Supported text"
    assert result.skipped == [str(unsupported)]
    assert result.errors == {}
