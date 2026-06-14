from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    source: str
    page: int | None = None
    chunk_index: int | None = None
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class IngestResponse(BaseModel):
    loaded_documents: int
    indexed_chunks: int
    skipped: list[str]
    errors: dict[str, str]


class EvaluationSummaryResponse(BaseModel):
    cases: int
    hit_rate_at_k: float
    mrr_at_k: float
    source_recall: float


class EvaluationCaseResponse(BaseModel):
    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    hit: bool


class EvaluationReportResponse(BaseModel):
    summary: EvaluationSummaryResponse
    cases: list[EvaluationCaseResponse]
