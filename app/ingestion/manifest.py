import fnmatch
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.ingestion.loaders import LoadedDocument


ACL_METADATA_KEYS = {
    "tenant_id",
    "allowed_user_ids",
    "allowed_department_ids",
    "allowed_roles",
    "is_public",
    "doc_id",
    "document_version",
}


@dataclass(frozen=True)
class DocumentManifestEntry:
    pattern: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentManifest:
    entries: tuple[DocumentManifestEntry, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> "DocumentManifest":
        if path is None or not path.exists():
            return cls()

        raw = json.loads(path.read_text(encoding="utf-8"))
        defaults: dict[str, Any] = {}
        entries: list[DocumentManifestEntry] = []

        if isinstance(raw, dict):
            defaults = dict(raw.get("defaults") or {})
            entries.extend(_entries_from_documents(raw.get("documents") or []))
            entries.extend(_entries_from_patterns(raw.get("patterns") or {}))
        elif isinstance(raw, list):
            entries.extend(_entries_from_documents(raw))

        return cls(entries=tuple(entries), defaults=defaults)

    def metadata_for_source(self, source: str, *, documents_dir: Path | None = None) -> dict[str, Any]:
        candidates = _source_candidates(source, documents_dir=documents_dir)
        metadata = dict(self.defaults)
        for entry in self.entries:
            if any(fnmatch.fnmatch(candidate, entry.pattern) for candidate in candidates):
                metadata.update(entry.metadata)
        return metadata


def apply_document_manifest(
    documents: list[LoadedDocument],
    *,
    manifest_path: Path | None,
    documents_dir: Path | None,
    default_tenant_id: str,
    default_document_version: str,
) -> list[LoadedDocument]:
    manifest = DocumentManifest.load(manifest_path)
    enriched: list[LoadedDocument] = []
    for document in documents:
        manifest_metadata = manifest.metadata_for_source(document.source, documents_dir=documents_dir)
        acl_metadata = build_acl_metadata(
            document.source,
            manifest_metadata=manifest_metadata,
            default_tenant_id=default_tenant_id,
            default_document_version=default_document_version,
        )
        metadata = dict(document.metadata)
        metadata.update(acl_metadata)
        enriched.append(
            LoadedDocument(
                text=document.text,
                source=document.source,
                metadata=metadata,
            )
        )
    return enriched


def build_acl_metadata(
    source: str,
    *,
    manifest_metadata: dict[str, Any],
    default_tenant_id: str,
    default_document_version: str,
) -> dict[str, Any]:
    metadata = {
        "tenant_id": str(manifest_metadata.get("tenant_id") or default_tenant_id),
        "doc_id": str(manifest_metadata.get("doc_id") or _stable_doc_id(source)),
        "document_version": str(manifest_metadata.get("document_version") or default_document_version),
        "allowed_user_ids": _json_list(manifest_metadata.get("allowed_user_ids")),
        "allowed_department_ids": _json_list(manifest_metadata.get("allowed_department_ids")),
        "allowed_roles": _json_list(manifest_metadata.get("allowed_roles")),
        "is_public": _bool_value(manifest_metadata.get("is_public", True)),
    }
    return metadata


def _entries_from_documents(rows: list[Any]) -> list[DocumentManifestEntry]:
    entries: list[DocumentManifestEntry] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pattern = row.get("pattern") or row.get("path") or row.get("local_path") or row.get("source")
        if not pattern:
            continue
        metadata = dict(row.get("acl") or {})
        for key in ACL_METADATA_KEYS:
            if key in row:
                metadata[key] = row[key]
        entries.append(DocumentManifestEntry(pattern=str(pattern), metadata=metadata))
    return entries


def _entries_from_patterns(patterns: dict[str, Any]) -> list[DocumentManifestEntry]:
    entries: list[DocumentManifestEntry] = []
    for pattern, metadata in patterns.items():
        if isinstance(metadata, dict):
            row = dict(metadata.get("acl") or metadata)
        else:
            row = {}
        entries.append(DocumentManifestEntry(pattern=str(pattern), metadata=row))
    return entries


def _source_candidates(source: str, *, documents_dir: Path | None) -> tuple[str, ...]:
    source_path = Path(source)
    source_posix = source_path.as_posix()
    candidates = {source_posix, source_path.name}
    if documents_dir is not None:
        try:
            candidates.add(source_path.relative_to(documents_dir).as_posix())
        except ValueError:
            pass
    try:
        candidates.add(source_path.resolve().as_posix())
    except OSError:
        pass
    return tuple(candidates)


def _json_list(value: Any) -> str:
    if value is None:
        values: list[str] = []
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                values = [str(item) for item in parsed]
            else:
                values = [item.strip() for item in stripped.split(",")]
        else:
            values = [item.strip() for item in stripped.split(",")]
    elif isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    return json.dumps([item for item in values if item], ensure_ascii=False)


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _stable_doc_id(source: str) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"
