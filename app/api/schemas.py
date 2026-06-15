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


class IngestResponse(BaseModel):
    loaded_documents: int
    indexed_chunks: int
    skipped: list[str]
    errors: dict[str, str]


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
    retrieved_sources: list[str]
    retrieved: list[EvaluationRetrievedDocumentResponse] = []
    hit: bool
    is_negative: bool = False


class EvaluationReportResponse(BaseModel):
    generated_at: str | None = None
    dataset_path: str | None = None
    variant: str | None = None
    top_k: int | None = None
    config: dict = {}
    summary: EvaluationSummaryResponse
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
    document_count: int
    chunk_count: int
    parent_chunk_count: int
    bm25_ready: bool
    chroma_collection_name: str
    bm25_corpus: JsonlCorpusStatusResponse
    parent_corpus: JsonlCorpusStatusResponse
    chroma: ChromaCorpusStatusResponse
