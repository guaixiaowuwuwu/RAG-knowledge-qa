from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")
    embedding_model: str = Field(default="bge-m3", alias="EMBEDDING_MODEL")
    documents_dir: Path = Field(default=Path("data/documents"), alias="DOCUMENTS_DIR")
    chroma_dir: Path = Field(default=Path("data/chroma"), alias="CHROMA_DIR")
    chroma_collection: str = Field(default="rag_knowledge_base", alias="CHROMA_COLLECTION")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=4, alias="RETRIEVAL_TOP_K")
    bm25_corpus_path: Path = Field(default=Path("data/chroma/bm25_corpus.jsonl"), alias="BM25_CORPUS_PATH")
    dense_retrieval_top_k: int = Field(default=20, alias="DENSE_RETRIEVAL_TOP_K")
    bm25_retrieval_top_k: int = Field(default=20, alias="BM25_RETRIEVAL_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    reranker_top_n: int = Field(default=5, alias="RERANKER_TOP_N")
    parent_corpus_path: Path = Field(default=Path("data/chroma/parent_corpus.jsonl"), alias="PARENT_CORPUS_PATH")
    parent_chunk_size: int = Field(default=2048, alias="PARENT_CHUNK_SIZE")
    parent_chunk_overlap: int = Field(default=160, alias="PARENT_CHUNK_OVERLAP")
    query_rewrite_enabled: bool = Field(default=True, alias="QUERY_REWRITE_ENABLED")
    hyde_enabled: bool = Field(default=True, alias="HYDE_ENABLED")
    max_query_variants: int = Field(default=4, alias="MAX_QUERY_VARIANTS")
    query_transform_timeout_seconds: float = Field(default=8.0, alias="QUERY_TRANSFORM_TIMEOUT_SECONDS")
    min_reranker_score: float = Field(default=-5.0, alias="MIN_RERANKER_SCORE")
    min_final_source_count: int = Field(default=1, alias="MIN_FINAL_SOURCE_COUNT")
    enable_low_confidence_refusal: bool = Field(default=True, alias="ENABLE_LOW_CONFIDENCE_REFUSAL")
    time_sensitive_refusal_enabled: bool = Field(default=True, alias="TIME_SENSITIVE_REFUSAL_ENABLED")
    eval_dataset_path: Path = Field(default=Path("data/eval/sample_eval.jsonl"), alias="EVAL_DATASET_PATH")
    max_question_chars: int = Field(default=2000, alias="MAX_QUESTION_CHARS")
    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    answer_cache_enabled: bool = Field(default=False, alias="ANSWER_CACHE_ENABLED")
    answer_cache_backend: str = Field(default="redis", alias="ANSWER_CACHE_BACKEND")
    answer_cache_ttl_seconds: int = Field(default=300, alias="ANSWER_CACHE_TTL_SECONDS")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    admin_api_keys: str = Field(default="", alias="ADMIN_API_KEYS")
    default_tenant_id: str = Field(default="default", alias="DEFAULT_TENANT_ID")
    audit_db_path: Path = Field(default=Path("data/runtime/audit.sqlite3"), alias="AUDIT_DB_PATH")
    audit_question_max_chars: int = Field(default=500, alias="AUDIT_QUESTION_MAX_CHARS")
    audit_answer_max_chars: int = Field(default=1000, alias="AUDIT_ANSWER_MAX_CHARS")
    permission_version: str = Field(default="local-v1", alias="PERMISSION_VERSION")
    documents_manifest_path: Path = Field(default=Path("data/documents_manifest.json"), alias="DOCUMENTS_MANIFEST_PATH")
    document_index_version: str = Field(default="local-index-v1", alias="DOCUMENT_INDEX_VERSION")
    versioned_indexing_enabled: bool = Field(default=False, alias="VERSIONED_INDEXING_ENABLED")
    index_root_dir: Path = Field(default=Path("data/indexes"), alias="INDEX_ROOT_DIR")
    active_index_version_path: Path = Field(default=Path("data/indexes/active_version.txt"), alias="ACTIVE_INDEX_VERSION_PATH")
    ingestion_mode: str = Field(default="sync", alias="INGESTION_MODE")
    permission_filter_overfetch_max: int = Field(default=100, alias="PERMISSION_FILTER_OVERFETCH_MAX")
    wecom_enabled: bool = Field(default=False, alias="WECOM_ENABLED")
    wecom_corp_id: str = Field(default="", alias="WECOM_CORP_ID")
    wecom_agent_id: str = Field(default="", alias="WECOM_AGENT_ID")
    wecom_secret: str = Field(default="", alias="WECOM_SECRET")
    wecom_token: str = Field(default="", alias="WECOM_TOKEN")
    wecom_encoding_aes_key: str = Field(default="", alias="WECOM_ENCODING_AES_KEY")
    wecom_callback_path: str = Field(default="/integrations/wecom/callback", alias="WECOM_CALLBACK_PATH")
    wecom_response_mode: str = Field(default="active", alias="WECOM_RESPONSE_MODE")
    wecom_user_mapping_path: Path = Field(default=Path("data/runtime/wecom_users.json"), alias="WECOM_USER_MAPPING_PATH")
    wecom_api_base_url: str = Field(default="https://qyapi.weixin.qq.com/cgi-bin", alias="WECOM_API_BASE_URL")
    wecom_request_timeout_seconds: float = Field(default=8.0, alias="WECOM_REQUEST_TIMEOUT_SECONDS")
    wecom_retry_count: int = Field(default=2, alias="WECOM_RETRY_COUNT")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def model_post_init(self, __context) -> None:
        if not self.versioned_indexing_enabled:
            return
        try:
            if self.active_index_version_path.exists():
                active_version = self.active_index_version_path.read_text(encoding="utf-8").strip()
                if active_version:
                    self.document_index_version = active_version
        except OSError:
            pass


@lru_cache
def get_settings() -> Settings:
    return Settings()
