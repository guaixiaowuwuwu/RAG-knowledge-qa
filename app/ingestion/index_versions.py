import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_INDEX_VERSION = "local-index-v1"


@dataclass(frozen=True)
class IndexPaths:
    version: str
    index_dir: Path
    chroma_dir: Path
    bm25_corpus_path: Path
    parent_corpus_path: Path


@dataclass(frozen=True)
class IndexVersionInfo:
    version: str
    index_dir: Path
    is_active: bool
    exists: bool
    bm25_count: int
    parent_count: int
    chroma_exists: bool
    updated_at: str | None = None


def versioned_indexing_enabled(settings) -> bool:
    return bool(getattr(settings, "versioned_indexing_enabled", False))


def get_index_root_dir(settings) -> Path:
    return Path(getattr(settings, "index_root_dir", Path("data/indexes")))


def get_active_index_version_path(settings) -> Path:
    default_path = get_index_root_dir(settings) / "active_version.txt"
    return Path(getattr(settings, "active_index_version_path", default_path))


def get_configured_index_version(settings) -> str:
    version = str(getattr(settings, "document_index_version", DEFAULT_INDEX_VERSION)).strip()
    return version or DEFAULT_INDEX_VERSION


def get_active_index_version(settings) -> str:
    if not versioned_indexing_enabled(settings):
        return get_configured_index_version(settings)
    active_path = get_active_index_version_path(settings)
    try:
        if active_path.exists():
            version = active_path.read_text(encoding="utf-8").strip()
            if version:
                return version
    except OSError:
        pass
    return get_configured_index_version(settings)


def get_index_paths(settings, version: str | None = None) -> IndexPaths:
    resolved_version = str(version or get_active_index_version(settings)).strip() or DEFAULT_INDEX_VERSION
    if not versioned_indexing_enabled(settings):
        chroma_dir = Path(getattr(settings, "chroma_dir", Path("data/chroma")))
        bm25_corpus_path = Path(getattr(settings, "bm25_corpus_path", chroma_dir / "bm25_corpus.jsonl"))
        parent_corpus_path = Path(getattr(settings, "parent_corpus_path", chroma_dir / "parent_corpus.jsonl"))
        return IndexPaths(
            version=resolved_version,
            index_dir=chroma_dir.parent,
            chroma_dir=chroma_dir,
            bm25_corpus_path=bm25_corpus_path,
            parent_corpus_path=parent_corpus_path,
        )

    index_dir = get_index_root_dir(settings) / resolved_version
    return IndexPaths(
        version=resolved_version,
        index_dir=index_dir,
        chroma_dir=index_dir / "chroma",
        bm25_corpus_path=index_dir / "bm25_corpus.jsonl",
        parent_corpus_path=index_dir / "parent_corpus.jsonl",
    )


def activate_index_version(settings, version: str, *, require_exists: bool = True) -> None:
    paths = get_index_paths(settings, version=version)
    if require_exists and not paths.index_dir.exists():
        raise FileNotFoundError(f"Index version does not exist: {version}")

    active_path = get_active_index_version_path(settings)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = active_path.with_name(f".{active_path.name}.tmp")
    temp_path.write_text(f"{paths.version}\n", encoding="utf-8")
    os.replace(temp_path, active_path)


def validate_index_build(paths: IndexPaths, result) -> None:
    indexed_chunks = int(getattr(result, "indexed_chunks", 0) or 0)
    if indexed_chunks <= 0:
        raise ValueError("Index build produced no chunks.")
    if not paths.chroma_dir.exists():
        raise ValueError(f"Chroma directory was not created: {paths.chroma_dir}")
    bm25_count = count_jsonl_rows(paths.bm25_corpus_path)
    if bm25_count <= 0:
        raise ValueError(f"BM25 corpus is empty or missing: {paths.bm25_corpus_path}")
    if not paths.parent_corpus_path.exists():
        raise ValueError(f"Parent corpus is missing: {paths.parent_corpus_path}")
    if count_jsonl_rows(paths.parent_corpus_path) <= 0:
        raise ValueError(f"Parent corpus is empty: {paths.parent_corpus_path}")


def list_index_versions(settings) -> list[IndexVersionInfo]:
    active_version = get_active_index_version(settings)
    if not versioned_indexing_enabled(settings):
        return [_index_version_info(get_index_paths(settings), active_version=active_version)]

    root_dir = get_index_root_dir(settings)
    if not root_dir.exists():
        return []

    versions = [
        _index_version_info(get_index_paths(settings, version=path.name), active_version=active_version)
        for path in sorted(root_dir.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ]
    return versions


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def generate_index_version(prefix: str = "index") -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{timestamp}"


def _index_version_info(paths: IndexPaths, *, active_version: str) -> IndexVersionInfo:
    updated_at = None
    if paths.index_dir.exists():
        modified_times = [path.stat().st_mtime for path in paths.index_dir.rglob("*") if path.exists()]
        if modified_times:
            updated_at = datetime.fromtimestamp(max(modified_times), UTC).isoformat()

    return IndexVersionInfo(
        version=paths.version,
        index_dir=paths.index_dir,
        is_active=paths.version == active_version,
        exists=paths.index_dir.exists(),
        bm25_count=count_jsonl_rows(paths.bm25_corpus_path),
        parent_count=count_jsonl_rows(paths.parent_corpus_path),
        chroma_exists=(paths.chroma_dir / "chroma.sqlite3").exists() or paths.chroma_dir.exists(),
        updated_at=updated_at,
    )
