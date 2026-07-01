from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from app.ingestion.index_versions import (
    IndexPaths,
    activate_index_version,
    generate_index_version,
    get_index_paths,
    validate_index_build,
)
from app.ingestion.pipeline import ingest_directory
from app.observability.metrics import record_ingestion_job_status
from app.rag.embeddings import build_embeddings
from app.rag.vector_store import ChromaVectorStore
from app.storage.sqlite import connect_sqlite


JOB_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")


@dataclass(frozen=True)
class IngestionJob:
    id: str
    tenant_id: str
    requested_by: str
    status: str
    input_path: str
    target_index_version: str
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class IngestionJobRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._ensure_schema()

    @classmethod
    def from_settings(cls, settings) -> "IngestionJobRepository":
        return cls(Path(getattr(settings, "audit_db_path", Path("data/runtime/audit.sqlite3"))))

    def create_job(
        self,
        *,
        tenant_id: str,
        requested_by: str,
        input_path: str | Path,
        target_index_version: str | None = None,
    ) -> IngestionJob:
        now = _now()
        job = IngestionJob(
            id=str(uuid4()),
            tenant_id=str(tenant_id),
            requested_by=str(requested_by),
            status="queued",
            input_path=str(input_path),
            target_index_version=str(target_index_version or generate_index_version()),
            created_at=now,
            updated_at=now,
        )
        with connect_sqlite(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    id, tenant_id, requested_by, status, input_path, target_index_version,
                    error, started_at, finished_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.tenant_id,
                    job.requested_by,
                    job.status,
                    job.input_path,
                    job.target_index_version,
                    job.error,
                    job.started_at,
                    job.finished_at,
                    job.created_at,
                    job.updated_at,
                ),
            )
        return job

    def get_job(self, job_id: str, *, tenant_id: str | None = None) -> IngestionJob | None:
        query = "SELECT * FROM ingestion_jobs WHERE id = ?"
        params: list[object] = [job_id]
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        with connect_sqlite(self.db_path) as connection:
            row = connection.execute(query, params).fetchone()
        return _job_from_row(row) if row is not None else None

    def list_jobs(self, *, tenant_id: str | None = None, limit: int = 50) -> list[IngestionJob]:
        query = "SELECT * FROM ingestion_jobs"
        params: list[object] = []
        if tenant_id is not None:
            query += " WHERE tenant_id = ?"
            params.append(tenant_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with connect_sqlite(self.db_path) as connection:
            rows = connection.execute(query, params).fetchall()
        return [_job_from_row(row) for row in rows]

    def next_queued_job(self, *, tenant_id: str | None = None) -> IngestionJob | None:
        query = "SELECT * FROM ingestion_jobs WHERE status = 'queued'"
        params: list[object] = []
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        query += " ORDER BY created_at ASC LIMIT 1"
        with connect_sqlite(self.db_path) as connection:
            row = connection.execute(query, params).fetchone()
        return _job_from_row(row) if row is not None else None

    def mark_running(self, job_id: str) -> None:
        self._update_status(job_id, "running", started_at=_now(), error=None)
        record_ingestion_job_status("running")

    def mark_succeeded(self, job_id: str) -> None:
        self._update_status(job_id, "succeeded", finished_at=_now(), error=None)
        record_ingestion_job_status("succeeded")

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update_status(job_id, "failed", finished_at=_now(), error=error[:2000])
        record_ingestion_job_status("failed")

    def mark_cancelled(self, job_id: str, error: str | None = None) -> None:
        self._update_status(job_id, "cancelled", finished_at=_now(), error=error)
        record_ingestion_job_status("cancelled")

    def latest_status_by_version(self, *, tenant_id: str | None = None) -> dict[str, str]:
        jobs = self.list_jobs(tenant_id=tenant_id, limit=500)
        statuses: dict[str, str] = {}
        for job in jobs:
            statuses.setdefault(job.target_index_version, job.status)
        return statuses

    def _ensure_schema(self) -> None:
        with connect_sqlite(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
                    input_path TEXT NOT NULL,
                    target_index_version TEXT NOT NULL,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_tenant_status ON ingestion_jobs (tenant_id, status, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_version ON ingestion_jobs (target_index_version)"
            )

    def _update_status(
        self,
        job_id: str,
        status: str,
        *,
        started_at: str | None = None,
        finished_at: str | None = None,
        error: str | None = None,
    ) -> None:
        if status not in JOB_STATUSES:
            raise ValueError(f"Unsupported ingestion job status: {status}")

        assignments = ["status = ?", "updated_at = ?"]
        params: list[object] = [status, _now()]
        if started_at is not None:
            assignments.append("started_at = ?")
            params.append(started_at)
        if finished_at is not None:
            assignments.append("finished_at = ?")
            params.append(finished_at)
        assignments.append("error = ?")
        params.append(error)
        params.append(job_id)

        with connect_sqlite(self.db_path) as connection:
            connection.execute(
                f"UPDATE ingestion_jobs SET {', '.join(assignments)} WHERE id = ?",
                params,
            )


def run_ingestion_job(
    settings,
    job_id: str,
    *,
    ingest_fn: Callable = ingest_directory,
    vector_store_factory: Callable[[IndexPaths, object], object] | None = None,
    activate_on_success: bool = True,
) -> IngestionJob | None:
    repository = IngestionJobRepository.from_settings(settings)
    job = repository.get_job(job_id)
    if job is None:
        return None
    if job.status not in {"queued", "running"}:
        return job

    repository.mark_running(job.id)
    paths = get_index_paths(settings, version=job.target_index_version)
    paths.index_dir.mkdir(parents=True, exist_ok=True)

    try:
        vector_store = (
            vector_store_factory(paths, settings)
            if vector_store_factory is not None
            else _default_vector_store_factory(paths, settings)
        )
        result = ingest_fn(
            documents_dir=Path(job.input_path),
            vector_store=vector_store,
            chunk_size=int(getattr(settings, "chunk_size", 800)),
            chunk_overlap=int(getattr(settings, "chunk_overlap", 120)),
            reset=True,
            bm25_corpus_path=paths.bm25_corpus_path,
            parent_corpus_path=paths.parent_corpus_path,
            parent_chunk_size=int(getattr(settings, "parent_chunk_size", 2048)),
            parent_chunk_overlap=int(getattr(settings, "parent_chunk_overlap", 160)),
            manifest_path=getattr(settings, "documents_manifest_path", None),
            default_tenant_id=str(getattr(settings, "default_tenant_id", "default")),
            document_index_version=paths.version,
        )
        validate_index_build(paths, result)
        repository.mark_succeeded(job.id)
        if activate_on_success:
            activate_index_version(settings, paths.version, require_exists=True)
    except Exception as exc:
        repository.mark_failed(job.id, f"{type(exc).__name__}: {exc}")

    return repository.get_job(job.id)


def process_next_job(
    settings,
    *,
    ingest_fn: Callable = ingest_directory,
    vector_store_factory: Callable[[IndexPaths, object], object] | None = None,
) -> IngestionJob | None:
    repository = IngestionJobRepository.from_settings(settings)
    job = repository.next_queued_job()
    if job is None:
        return None
    return run_ingestion_job(
        settings,
        job.id,
        ingest_fn=ingest_fn,
        vector_store_factory=vector_store_factory,
    )


def _default_vector_store_factory(paths: IndexPaths, settings):
    return ChromaVectorStore(
        persist_dir=paths.chroma_dir,
        collection_name=str(getattr(settings, "chroma_collection", "rag_knowledge_base")),
        embeddings=build_embeddings(settings),
    )


def _job_from_row(row) -> IngestionJob:
    return IngestionJob(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        requested_by=str(row["requested_by"]),
        status=str(row["status"]),
        input_path=str(row["input_path"]),
        target_index_version=str(row["target_index_version"]),
        error=row["error"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
