from app.core.config import Settings


DEFAULT_ENV_KEYS = [
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "RERANKER_TOP_N",
    "PARENT_CORPUS_PATH",
    "PARENT_CHUNK_SIZE",
    "PARENT_CHUNK_OVERLAP",
    "QUERY_REWRITE_ENABLED",
    "HYDE_ENABLED",
    "MAX_QUERY_VARIANTS",
    "QUERY_TRANSFORM_TIMEOUT_SECONDS",
    "MIN_RERANKER_SCORE",
    "MIN_FINAL_SOURCE_COUNT",
    "ENABLE_LOW_CONFIDENCE_REFUSAL",
    "TIME_SENSITIVE_REFUSAL_ENABLED",
    "EVAL_DATASET_PATH",
    "MAX_QUESTION_CHARS",
    "LLM_TIMEOUT_SECONDS",
    "ANSWER_CACHE_ENABLED",
    "ANSWER_CACHE_BACKEND",
    "ANSWER_CACHE_TTL_SECONDS",
    "REDIS_URL",
    "AUTH_ENABLED",
    "ADMIN_API_KEYS",
    "DEFAULT_TENANT_ID",
    "AUDIT_DB_PATH",
    "AUDIT_QUESTION_MAX_CHARS",
    "AUDIT_ANSWER_MAX_CHARS",
    "PERMISSION_VERSION",
    "DOCUMENTS_MANIFEST_PATH",
    "DOCUMENT_INDEX_VERSION",
    "VERSIONED_INDEXING_ENABLED",
    "INDEX_ROOT_DIR",
    "ACTIVE_INDEX_VERSION_PATH",
    "PERMISSION_FILTER_OVERFETCH_MAX",
]


def settings_without_env(monkeypatch) -> Settings:
    for key in DEFAULT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None)


def test_phase5_embedding_default_is_local_bge_m3(monkeypatch):
    settings = settings_without_env(monkeypatch)

    assert settings.embedding_model == "bge-m3"


def test_phase5_advanced_defaults(monkeypatch):
    settings = settings_without_env(monkeypatch)

    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker_top_n == 5
    assert settings.parent_corpus_path.as_posix() == "data/chroma/parent_corpus.jsonl"
    assert settings.parent_chunk_size == 2048
    assert settings.parent_chunk_overlap == 160
    assert settings.query_rewrite_enabled is True
    assert settings.hyde_enabled is True
    assert settings.max_query_variants == 4
    assert settings.query_transform_timeout_seconds == 8.0
    assert settings.min_reranker_score == -5.0
    assert settings.min_final_source_count == 1
    assert settings.enable_low_confidence_refusal is True
    assert settings.time_sensitive_refusal_enabled is True
    assert settings.eval_dataset_path.as_posix() == "data/eval/sample_eval.jsonl"
    assert settings.max_question_chars == 2000
    assert settings.llm_timeout_seconds == 60.0
    assert settings.answer_cache_enabled is False
    assert settings.answer_cache_backend == "redis"
    assert settings.answer_cache_ttl_seconds == 300
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.auth_enabled is False
    assert settings.admin_api_keys == ""
    assert settings.default_tenant_id == "default"
    assert settings.audit_db_path.as_posix() == "data/runtime/audit.sqlite3"
    assert settings.audit_question_max_chars == 500
    assert settings.audit_answer_max_chars == 1000
    assert settings.permission_version == "local-v1"
    assert settings.documents_manifest_path.as_posix() == "data/documents_manifest.json"
    assert settings.document_index_version == "local-index-v1"
    assert settings.versioned_indexing_enabled is False
    assert settings.permission_filter_overfetch_max == 100


def test_phase5_ragas_uses_business_judge_and_local_embedding(monkeypatch):
    settings = settings_without_env(monkeypatch)

    assert not hasattr(settings, "ragas_api_key")
    assert not hasattr(settings, "ragas_base_url")
    assert not hasattr(settings, "ragas_judge_model")
    assert not hasattr(settings, "ragas_embedding_model")
