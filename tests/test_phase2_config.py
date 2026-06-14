from app.core.config import Settings


def test_phase2_settings_defaults():
    settings = Settings()

    assert settings.bm25_corpus_path.as_posix() == "data/chroma/bm25_corpus.jsonl"
    assert settings.dense_retrieval_top_k == 20
    assert settings.bm25_retrieval_top_k == 20
    assert settings.rrf_k == 60
    assert settings.reranker_enabled is False
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker_top_n == 5
