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
    eval_dataset_path: Path = Field(default=Path("data/eval/sample_eval.jsonl"), alias="EVAL_DATASET_PATH")
    max_question_chars: int = Field(default=2000, alias="MAX_QUESTION_CHARS")
    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    answer_cache_enabled: bool = Field(default=False, alias="ANSWER_CACHE_ENABLED")
    answer_cache_backend: str = Field(default="redis", alias="ANSWER_CACHE_BACKEND")
    answer_cache_ttl_seconds: int = Field(default=300, alias="ANSWER_CACHE_TTL_SECONDS")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
