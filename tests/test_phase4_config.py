from app.core.config import Settings


def test_phase4_settings_defaults():
    settings = Settings()

    assert settings.parent_corpus_path.as_posix() == "data/chroma/parent_corpus.jsonl"
    assert settings.parent_chunk_size == 2048
    assert settings.parent_chunk_overlap == 160
    assert settings.query_rewrite_enabled is True
    assert settings.hyde_enabled is True
    assert settings.max_query_variants == 4
    assert settings.eval_dataset_path.as_posix() == "data/eval/sample_eval.jsonl"
