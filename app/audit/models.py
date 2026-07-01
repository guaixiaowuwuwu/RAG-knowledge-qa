from dataclasses import dataclass, field


@dataclass(frozen=True)
class AuditSource:
    source_path: str
    page: int | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    document_version: str | None = None


@dataclass(frozen=True)
class AuditSession:
    id: str
    tenant_id: str
    user_id_hash: str
    source: str
    index_version: str | None
    question_hash: str
    redacted_question: str
    answer_summary: str
    refusal_reason: str | None
    latency_ms: float | None
    token_usage: dict | None
    created_at: str


@dataclass(frozen=True)
class AuditFeedback:
    id: int
    session_id: str
    tenant_id: str
    user_id_hash: str
    rating: int
    tags: tuple[str, ...] = field(default_factory=tuple)
    comment: str | None = None
    created_at: str = ""


@dataclass(frozen=True)
class AuditSessionDetail:
    session: AuditSession
    sources: list[AuditSource]
    feedback: list[AuditFeedback]
