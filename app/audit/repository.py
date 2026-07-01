import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.audit.models import AuditFeedback, AuditSession, AuditSessionDetail, AuditSource
from app.ingestion.index_versions import get_active_index_version
from app.security.context import RequestContext
from app.storage.sqlite import connect_sqlite


class AuditRepository(Protocol):
    def record_qa_session(
        self,
        *,
        question: str,
        answer: str,
        sources: list[object],
        context: RequestContext | None,
        refusal_reason: str | None,
        latency_ms: float | None,
        token_usage: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        ...

    def list_sessions(self, *, tenant_id: str, limit: int = 50, offset: int = 0) -> list[AuditSession]:
        ...

    def get_session(self, session_id: str, *, tenant_id: str) -> AuditSessionDetail | None:
        ...

    def add_feedback(
        self,
        *,
        session_id: str,
        context: RequestContext,
        rating: int,
        tags: list[str] | tuple[str, ...] | None = None,
        comment: str | None = None,
    ) -> int | None:
        ...


class NullAuditRepository:
    def record_qa_session(
        self,
        *,
        question: str,
        answer: str,
        sources: list[object],
        context: RequestContext | None,
        refusal_reason: str | None,
        latency_ms: float | None,
        token_usage: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        return session_id or uuid.uuid4().hex

    def list_sessions(self, *, tenant_id: str, limit: int = 50, offset: int = 0) -> list[AuditSession]:
        return []

    def get_session(self, session_id: str, *, tenant_id: str) -> AuditSessionDetail | None:
        return None

    def add_feedback(
        self,
        *,
        session_id: str,
        context: RequestContext,
        rating: int,
        tags: list[str] | tuple[str, ...] | None = None,
        comment: str | None = None,
    ) -> int | None:
        return None


class SqliteAuditRepository:
    def __init__(
        self,
        path: str | Path,
        *,
        question_max_chars: int = 500,
        answer_max_chars: int = 1000,
        index_version: str | None = None,
    ):
        self.path = Path(path)
        self.question_max_chars = question_max_chars
        self.answer_max_chars = answer_max_chars
        self.index_version = index_version
        self.ensure_schema()

    @classmethod
    def from_settings(cls, settings) -> "SqliteAuditRepository":
        return cls(
            path=getattr(settings, "audit_db_path", "data/runtime/audit.sqlite3"),
            question_max_chars=int(getattr(settings, "audit_question_max_chars", 500)),
            answer_max_chars=int(getattr(settings, "audit_answer_max_chars", 1000)),
            index_version=_index_version_from_settings(settings),
        )

    def ensure_schema(self) -> None:
        with connect_sqlite(self.path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS qa_sessions (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id_hash TEXT NOT NULL,
                    source TEXT NOT NULL,
                    index_version TEXT,
                    question_hash TEXT NOT NULL,
                    redacted_question TEXT NOT NULL,
                    answer_summary TEXT NOT NULL,
                    refusal_reason TEXT,
                    latency_ms REAL,
                    token_usage TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_qa_sessions_tenant_created
                ON qa_sessions (tenant_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS qa_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    page INTEGER,
                    chunk_id TEXT,
                    chunk_index INTEGER,
                    document_version TEXT,
                    FOREIGN KEY(session_id) REFERENCES qa_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_qa_sources_session
                ON qa_sources (session_id);

                CREATE TABLE IF NOT EXISTS qa_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id_hash TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    comment TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES qa_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_qa_feedback_session
                ON qa_feedback (session_id);
                """
            )
            _ensure_column(connection, "qa_sessions", "index_version", "TEXT")

    def record_qa_session(
        self,
        *,
        question: str,
        answer: str,
        sources: list[object],
        context: RequestContext | None,
        refusal_reason: str | None,
        latency_ms: float | None,
        token_usage: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        created_at = _utc_now()
        session_id = session_id or uuid.uuid4().hex
        tenant_id = getattr(context, "tenant_id", "unknown") if context is not None else "unknown"
        user_id = getattr(context, "user_id", "") if context is not None else ""
        source = getattr(context, "source", "unknown") if context is not None else "unknown"
        audit_sources = [_source_from_answer_source(source_item) for source_item in sources]

        with connect_sqlite(self.path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                INSERT INTO qa_sessions (
                    id,
                    tenant_id,
                    user_id_hash,
                    source,
                    index_version,
                    question_hash,
                    redacted_question,
                    answer_summary,
                    refusal_reason,
                    latency_ms,
                    token_usage,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(tenant_id),
                    hash_identifier(str(user_id)),
                    str(source),
                    self.index_version,
                    hash_text(question),
                    truncate_text(question, self.question_max_chars),
                    truncate_text(answer, self.answer_max_chars),
                    refusal_reason,
                    latency_ms,
                    json.dumps(token_usage, ensure_ascii=False, sort_keys=True) if token_usage else None,
                    created_at,
                ),
            )
            connection.executemany(
                """
                INSERT INTO qa_sources (
                    session_id,
                    tenant_id,
                    source_path,
                    page,
                    chunk_id,
                    chunk_index,
                    document_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        str(tenant_id),
                        audit_source.source_path,
                        audit_source.page,
                        audit_source.chunk_id,
                        audit_source.chunk_index,
                        audit_source.document_version,
                    )
                    for audit_source in audit_sources
                ],
            )
        return session_id

    def list_sessions(self, *, tenant_id: str, limit: int = 50, offset: int = 0) -> list[AuditSession]:
        with connect_sqlite(self.path) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM qa_sessions
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (tenant_id, max(1, min(int(limit), 200)), max(0, int(offset))),
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def get_session(self, session_id: str, *, tenant_id: str) -> AuditSessionDetail | None:
        with connect_sqlite(self.path) as connection:
            session_row = connection.execute(
                """
                SELECT *
                FROM qa_sessions
                WHERE id = ? AND tenant_id = ?
                """,
                (session_id, tenant_id),
            ).fetchone()
            if session_row is None:
                return None
            source_rows = connection.execute(
                """
                SELECT source_path, page, chunk_id, chunk_index, document_version
                FROM qa_sources
                WHERE session_id = ? AND tenant_id = ?
                ORDER BY id ASC
                """,
                (session_id, tenant_id),
            ).fetchall()
            feedback_rows = connection.execute(
                """
                SELECT *
                FROM qa_feedback
                WHERE session_id = ? AND tenant_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id, tenant_id),
            ).fetchall()
        return AuditSessionDetail(
            session=_session_from_row(session_row),
            sources=[_source_from_row(row) for row in source_rows],
            feedback=[_feedback_from_row(row) for row in feedback_rows],
        )

    def add_feedback(
        self,
        *,
        session_id: str,
        context: RequestContext,
        rating: int,
        tags: list[str] | tuple[str, ...] | None = None,
        comment: str | None = None,
    ) -> int | None:
        tenant_id = context.tenant_id
        with connect_sqlite(self.path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            session_row = connection.execute(
                """
                SELECT id
                FROM qa_sessions
                WHERE id = ? AND tenant_id = ?
                """,
                (session_id, tenant_id),
            ).fetchone()
            if session_row is None:
                return None
            cursor = connection.execute(
                """
                INSERT INTO qa_feedback (
                    session_id,
                    tenant_id,
                    user_id_hash,
                    rating,
                    tags,
                    comment,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tenant_id,
                    hash_identifier(context.user_id),
                    rating,
                    json.dumps(list(tags or []), ensure_ascii=False),
                    truncate_text(comment, 1000) if comment else None,
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)


def build_audit_repository(settings) -> AuditRepository:
    if not hasattr(settings, "audit_db_path"):
        return NullAuditRepository()
    return SqliteAuditRepository.from_settings(settings)


def _session_from_row(row) -> AuditSession:
    token_usage_raw = row["token_usage"]
    token_usage = None
    if token_usage_raw:
        try:
            token_usage = json.loads(token_usage_raw)
        except json.JSONDecodeError:
            token_usage = None
    return AuditSession(
        id=row["id"],
        tenant_id=row["tenant_id"],
        user_id_hash=row["user_id_hash"],
        source=row["source"],
        index_version=row["index_version"],
        question_hash=row["question_hash"],
        redacted_question=row["redacted_question"],
        answer_summary=row["answer_summary"],
        refusal_reason=row["refusal_reason"],
        latency_ms=row["latency_ms"],
        token_usage=token_usage,
        created_at=row["created_at"],
    )


def _source_from_row(row) -> AuditSource:
    return AuditSource(
        source_path=row["source_path"],
        page=row["page"],
        chunk_id=row["chunk_id"],
        chunk_index=row["chunk_index"],
        document_version=row["document_version"],
    )


def _feedback_from_row(row) -> AuditFeedback:
    tags_raw = row["tags"]
    tags = []
    if tags_raw:
        try:
            parsed = json.loads(tags_raw)
            if isinstance(parsed, list):
                tags = [str(item) for item in parsed]
        except json.JSONDecodeError:
            tags = []
    return AuditFeedback(
        id=row["id"],
        session_id=row["session_id"],
        tenant_id=row["tenant_id"],
        user_id_hash=row["user_id_hash"],
        rating=row["rating"],
        tags=tuple(tags),
        comment=row["comment"],
        created_at=row["created_at"],
    )


def _source_from_answer_source(source: object) -> AuditSource:
    metadata = getattr(source, "metadata", {}) or {}
    source_path = getattr(source, "source", None) or metadata.get("source") or metadata.get("source_path") or ""
    return AuditSource(
        source_path=str(source_path),
        page=_optional_int(getattr(source, "page", None) if hasattr(source, "page") else metadata.get("page")),
        chunk_id=_optional_str(metadata.get("chunk_id") or metadata.get("id")),
        chunk_index=_optional_int(
            getattr(source, "chunk_index", None) if hasattr(source, "chunk_index") else metadata.get("chunk_index")
        ),
        document_version=_optional_str(metadata.get("document_version")),
    )


def hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16] if value else ""


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def truncate_text(value: str | None, max_chars: int) -> str:
    text = str(value or "").replace("\x00", "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    suffix = "..."
    if max_chars <= len(suffix):
        return text[:max_chars]
    return text[: max_chars - len(suffix)].rstrip() + suffix


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _index_version_from_settings(settings) -> str | None:
    try:
        return get_active_index_version(settings)
    except Exception:
        version = getattr(settings, "document_index_version", None)
        return str(version) if version is not None else None


def _ensure_column(connection, table_name: str, column_name: str, column_definition: str) -> None:
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
