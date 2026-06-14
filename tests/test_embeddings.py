from types import SimpleNamespace

from app.rag import embeddings


def test_build_embeddings_uses_local_bge_m3(monkeypatch):
    created = {}

    class FakeLocalEmbeddings:
        def __init__(self, model_name: str):
            created["model_name"] = model_name

    monkeypatch.setattr(embeddings, "LocalSentenceTransformerEmbeddings", FakeLocalEmbeddings)

    result = embeddings.build_embeddings(SimpleNamespace(embedding_model="bge-m3"))

    assert isinstance(result, FakeLocalEmbeddings)
    assert created["model_name"] == "bge-m3"
