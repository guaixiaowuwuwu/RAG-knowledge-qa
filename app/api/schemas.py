from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    debug: bool = False
    rewrite_enabled: bool | None = None
    hyde_enabled: bool | None = None
    parent_hydration_enabled: bool | None = None
    max_query_variants: int | None = Field(default=None, ge=1, le=8)


class SourceResponse(BaseModel):
    source: str
    page: int | None = None
    chunk_index: int | None = None
    matched_child_chunk_index: int | None = None
    content_type: str | None = None
    table_index: int | None = None
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    debug: dict | None = None
    session_id: str | None = None


class IngestResponse(BaseModel):
    loaded_documents: int
    indexed_chunks: int
    skipped: list[str]
    errors: dict[str, str]
    job_id: str | None = None
    status: str | None = None
    target_index_version: str | None = None


class EvaluationSummaryResponse(BaseModel):
    cases: int
    positive_cases: int = 0
    negative_cases: int = 0
    hit_rate_at_k: float
    mrr_at_k: float
    source_recall: float
    source_recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    negative_rejection_rate: float = 0.0
    page_hit_rate_at_k: float = 0.0
    evidence_keyword_recall_at_k: float = 0.0
    evidence_strict_hit_at_k: float = 0.0
    refusal_reasons: dict[str, int] = {}


class EvaluationRetrievedDocumentResponse(BaseModel):
    id: str
    source: str
    score: float | None = None
    page: int | None = None
    chunk_index: int | None = None
    matched_child_chunk_index: int | None = None
    content: str


class EvaluationCaseResponse(BaseModel):
    id: str = ""
    question: str
    ground_truth: str = ""
    expected_sources: list[str]
    expected_answer_keywords: list[str] = []
    expected_pages: dict[str, list[int]] = {}
    expected_chunk_keywords: list[str] = []
    evidence_notes: str = ""
    category: str = ""
    difficulty: str = ""
    language: str = ""
    retrieved_sources: list[str]
    retrieved: list[EvaluationRetrievedDocumentResponse] = []
    hit: bool
    is_negative: bool = False
    page_hit: bool | None = None
    evidence_keyword_matches: list[str] = []
    evidence_keyword_misses: list[str] = []
    evidence_keyword_recall: float | None = None
    evidence_strict_hit: bool | None = None
    refusal_reason: str | None = None
    confidence: dict = {}


class EvaluationReportResponse(BaseModel):
    generated_at: str | None = None
    dataset_path: str | None = None
    variant: str | None = None
    top_k: int | None = None
    config: dict = {}
    summary: EvaluationSummaryResponse
    groups: dict[str, dict[str, EvaluationSummaryResponse]] = {}
    cases: list[EvaluationCaseResponse]


class AnswerEvaluationReportResponse(BaseModel):
    available: bool
    report_path: str | None = None
    generated_at: str | None = None
    dataset_path: str | None = None
    top_k: int | None = None
    summary: dict = {}
    ragas: dict | None = None
    error: str | None = None


class RetrievalComparisonReportResponse(BaseModel):
    available: bool
    report_path: str | None = None
    generated_at: str | None = None
    dataset_path: str | None = None
    top_k: int | None = None
    variants: list[dict] = []
    table: list[dict] | str | None = None
    error: str | None = None


class JsonlCorpusStatusResponse(BaseModel):
    path: str
    exists: bool
    count: int = 0
    size_bytes: int = 0
    updated_at: str | None = None


class ChromaCorpusStatusResponse(BaseModel):
    persist_dir: str
    collection_name: str
    exists: bool
    chunk_count: int = 0
    size_bytes: int = 0
    updated_at: str | None = None
    error: str | None = None


class CorpusStatusResponse(BaseModel):
    documents_dir: str
    active_index_version: str | None = None
    index_dir: str | None = None
    ready: bool
    readiness_reason: str | None = None
    document_count: int
    chunk_count: int
    parent_chunk_count: int
    bm25_ready: bool
    chroma_collection_name: str
    bm25_corpus: JsonlCorpusStatusResponse
    parent_corpus: JsonlCorpusStatusResponse
    chroma: ChromaCorpusStatusResponse


class AuditSessionResponse(BaseModel):
    id: str
    tenant_id: str
    user_id_hash: str
    source: str
    index_version: str | None = None
    question_hash: str
    redacted_question: str
    answer_summary: str
    refusal_reason: str | None = None
    latency_ms: float | None = None
    token_usage: dict | None = None
    created_at: str


class AuditSourceResponse(BaseModel):
    source_path: str
    page: int | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    document_version: str | None = None


class AuditFeedbackResponse(BaseModel):
    id: int
    session_id: str
    tenant_id: str
    user_id_hash: str
    rating: int
    tags: list[str] = Field(default_factory=list)
    comment: str | None = None
    created_at: str


class AuditSessionListResponse(BaseModel):
    sessions: list[AuditSessionResponse]


class AuditSessionDetailResponse(BaseModel):
    session: AuditSessionResponse
    sources: list[AuditSourceResponse]
    feedback: list[AuditFeedbackResponse]


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    rating: int = Field(ge=-1, le=1)
    tags: list[str] = Field(default_factory=list, max_length=10)
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    feedback_id: int
    session_id: str


class IngestionJobCreateRequest(BaseModel):
    input_path: str | None = None
    target_index_version: str | None = None


class IngestionJobResponse(BaseModel):
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


class IndexVersionResponse(BaseModel):
    version: str
    index_dir: str
    is_active: bool
    exists: bool
    bm25_count: int
    parent_count: int
    chroma_exists: bool
    updated_at: str | None = None
    job_status: str | None = None


class IndexVersionListResponse(BaseModel):
    active_version: str
    indexes: list[IndexVersionResponse]


class IndexActivationResponse(BaseModel):
    version: str
    active_version: str
