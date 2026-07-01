from pathlib import Path
from types import SimpleNamespace

from app.core.config import Settings
from app.ingestion.index_versions import (
    activate_index_version,
    get_active_index_version,
    get_index_paths,
    list_index_versions,
    versioned_indexing_enabled,
)


def _settings(tmp_path: Path, **overrides):
    values = {
        "index_root_dir": tmp_path / "indexes",
        "active_index_version_path": tmp_path / "indexes" / "active_version.txt",
        "document_index_version": "configured-v1",
        "versioned_indexing_enabled": True,
        "chroma_dir": tmp_path / "legacy" / "chroma",
        "bm25_corpus_path": tmp_path / "legacy" / "bm25_corpus.jsonl",
        "parent_corpus_path": tmp_path / "legacy" / "parent_corpus.jsonl",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_settings_reads_document_index_version_from_active_file(tmp_path: Path):
    active_path = tmp_path / "indexes" / "active_version.txt"
    active_path.parent.mkdir(parents=True)
    active_path.write_text("active-v2\n", encoding="utf-8")

    settings = Settings(
        _env_file=None,
        VERSIONED_INDEXING_ENABLED=True,
        INDEX_ROOT_DIR=tmp_path / "indexes",
        ACTIVE_INDEX_VERSION_PATH=active_path,
        DOCUMENT_INDEX_VERSION="configured-v1",
    )

    assert settings.document_index_version == "active-v2"


def test_settings_ignores_active_file_when_versioning_disabled(tmp_path: Path):
    active_path = tmp_path / "indexes" / "active_version.txt"
    active_path.parent.mkdir(parents=True)
    active_path.write_text("active-v2\n", encoding="utf-8")

    settings = Settings(
        _env_file=None,
        VERSIONED_INDEXING_ENABLED=False,
        INDEX_ROOT_DIR=tmp_path / "indexes",
        ACTIVE_INDEX_VERSION_PATH=active_path,
        DOCUMENT_INDEX_VERSION="configured-v1",
    )

    assert settings.document_index_version == "configured-v1"
    assert get_active_index_version(settings) == "configured-v1"


def test_versioned_indexing_is_disabled_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VERSIONED_INDEXING_ENABLED", raising=False)
    settings = Settings(
        _env_file=None,
        CHROMA_DIR=tmp_path / "legacy" / "chroma",
        BM25_CORPUS_PATH=tmp_path / "legacy" / "bm25_corpus.jsonl",
        PARENT_CORPUS_PATH=tmp_path / "legacy" / "parent_corpus.jsonl",
        INDEX_ROOT_DIR=tmp_path / "indexes",
        ACTIVE_INDEX_VERSION_PATH=tmp_path / "indexes" / "active_version.txt",
        DOCUMENT_INDEX_VERSION="configured-v1",
    )

    paths = get_index_paths(settings)

    assert settings.versioned_indexing_enabled is False
    assert versioned_indexing_enabled(settings) is False
    assert paths.version == "configured-v1"
    assert paths.chroma_dir == settings.chroma_dir
    assert paths.bm25_corpus_path == settings.bm25_corpus_path
    assert paths.parent_corpus_path == settings.parent_corpus_path


def test_get_index_paths_uses_versioned_paths_when_enabled(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        VERSIONED_INDEXING_ENABLED=True,
        INDEX_ROOT_DIR=tmp_path / "indexes",
        ACTIVE_INDEX_VERSION_PATH=tmp_path / "indexes" / "active_version.txt",
        DOCUMENT_INDEX_VERSION="configured-v1",
    )

    paths = get_index_paths(settings)

    assert paths.chroma_dir == tmp_path / "indexes" / "configured-v1" / "chroma"
    assert paths.bm25_corpus_path == tmp_path / "indexes" / "configured-v1" / "bm25_corpus.jsonl"
    assert paths.parent_corpus_path == tmp_path / "indexes" / "configured-v1" / "parent_corpus.jsonl"


def test_active_index_version_file_overrides_configured_version(tmp_path: Path):
    settings = _settings(tmp_path)
    settings.active_index_version_path.parent.mkdir(parents=True)
    settings.active_index_version_path.write_text("active-v2\n", encoding="utf-8")

    paths = get_index_paths(settings)

    assert get_active_index_version(settings) == "active-v2"
    assert paths.version == "active-v2"
    assert paths.index_dir == tmp_path / "indexes" / "active-v2"
    assert paths.chroma_dir == paths.index_dir / "chroma"
    assert paths.bm25_corpus_path == paths.index_dir / "bm25_corpus.jsonl"
    assert paths.parent_corpus_path == paths.index_dir / "parent_corpus.jsonl"


def test_activate_previous_version_changes_retrieval_paths(tmp_path: Path):
    settings = _settings(tmp_path)
    (tmp_path / "indexes" / "v1" / "chroma").mkdir(parents=True)
    (tmp_path / "indexes" / "v1" / "bm25_corpus.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")
    (tmp_path / "indexes" / "v1" / "parent_corpus.jsonl").write_text('{"id":"p1"}\n', encoding="utf-8")
    (tmp_path / "indexes" / "v2" / "chroma").mkdir(parents=True)
    (tmp_path / "indexes" / "v2" / "bm25_corpus.jsonl").write_text('{"id":"2"}\n', encoding="utf-8")
    (tmp_path / "indexes" / "v2" / "parent_corpus.jsonl").write_text('{"id":"p2"}\n', encoding="utf-8")

    activate_index_version(settings, "v1")
    v1_paths = get_index_paths(settings)
    activate_index_version(settings, "v2")
    v2_paths = get_index_paths(settings)

    assert v1_paths.bm25_corpus_path == tmp_path / "indexes" / "v1" / "bm25_corpus.jsonl"
    assert v2_paths.bm25_corpus_path == tmp_path / "indexes" / "v2" / "bm25_corpus.jsonl"
    assert get_active_index_version(settings) == "v2"


def test_list_index_versions_marks_active_and_counts_artifacts(tmp_path: Path):
    settings = _settings(tmp_path)
    index_dir = tmp_path / "indexes" / "v1"
    (index_dir / "chroma").mkdir(parents=True)
    (index_dir / "chroma" / "chroma.sqlite3").write_text("fake", encoding="utf-8")
    (index_dir / "bm25_corpus.jsonl").write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    (index_dir / "parent_corpus.jsonl").write_text('{"id":"p1"}\n', encoding="utf-8")
    activate_index_version(settings, "v1")

    versions = list_index_versions(settings)

    assert len(versions) == 1
    assert versions[0].version == "v1"
    assert versions[0].is_active is True
    assert versions[0].bm25_count == 2
    assert versions[0].parent_count == 1
    assert versions[0].chroma_exists is True
