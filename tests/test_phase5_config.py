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
    "EVAL_DATASET_PATH",
    "MAX_QUESTION_CHARS",
    "LLM_TIMEOUT_SECONDS",
    "ANSWER_CACHE_ENABLED",
    "ANSWER_CACHE_BACKEND",
    "ANSWER_CACHE_TTL_SECONDS",
    "REDIS_URL",
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
    assert settings.eval_dataset_path.as_posix() == "data/eval/sample_eval.jsonl"
    assert settings.max_question_chars == 2000
    assert settings.llm_timeout_seconds == 60.0
    assert settings.answer_cache_enabled is False
    assert settings.answer_cache_backend == "redis"
    assert settings.answer_cache_ttl_seconds == 300
    assert settings.redis_url == "redis://localhost:6379/0"


def test_phase5_ragas_uses_business_judge_and_local_embedding(monkeypatch):
    settings = settings_without_env(monkeypatch)

    assert not hasattr(settings, "ragas_api_key")
    assert not hasattr(settings, "ragas_base_url")
    assert not hasattr(settings, "ragas_judge_model")
    assert not hasattr(settings, "ragas_embedding_model")
