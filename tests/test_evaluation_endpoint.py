import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app


def test_evaluation_report_endpoint(monkeypatch):
    from app.api import routes

    monkeypatch.setattr(
        routes,
        "build_evaluation_report",
        lambda: {"summary": {"cases": 1, "hit_rate_at_k": 1.0, "mrr_at_k": 1.0, "source_recall": 1.0}, "cases": []},
    )

    client = TestClient(app)
    response = client.get("/evaluation/report")

    assert response.status_code == 200
    assert response.json()["summary"]["hit_rate_at_k"] == 1.0


def test_answer_evaluation_report_endpoint(monkeypatch):
    from app.api import routes

    monkeypatch.setattr(
        routes,
        "build_latest_answer_evaluation_report",
        lambda: {
            "available": True,
            "report_path": "reports/answer-eval.json",
            "generated_at": "2026-06-14T00:00:00",
            "dataset_path": "data/eval/sample_eval.jsonl",
            "top_k": 4,
            "summary": {"cases": 2},
            "ragas": {
                "metrics": {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.8,
                    "context_precision": 0.75,
                    "context_recall": 0.82,
                },
                "target_thresholds": {"faithfulness": 0.85},
            },
        },
    )

    client = TestClient(app)
    response = client.get("/evaluation/answer-report")

    assert response.status_code == 200
    assert response.json()["ragas"]["metrics"]["faithfulness"] == 0.9


def test_retrieval_comparison_report_endpoint(monkeypatch):
    from app.api import routes

    monkeypatch.setattr(
        routes,
        "build_latest_retrieval_comparison_report",
        lambda: {
            "available": True,
            "report_path": "reports/retrieval-comparison.json",
            "dataset_path": "data/eval/sample_eval.jsonl",
            "top_k": 5,
            "variants": [
                {
                    "variant": "dense",
                    "summary": {"cases": 2, "hit_rate_at_k": 0.5},
                    "groups": {"language": {"zh": {"cases": 1, "hit_rate_at_k": 1.0}}},
                },
                {
                    "variant": "hybrid",
                    "summary": {"cases": 2, "hit_rate_at_k": 1.0},
                    "groups": {"language": {"zh": {"cases": 1, "hit_rate_at_k": 1.0}}},
                },
            ],
        },
    )

    client = TestClient(app)
    response = client.get("/evaluation/comparison-report")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["variants"][1]["variant"] == "hybrid"
    assert payload["variants"][0]["groups"]["language"]["zh"]["hit_rate_at_k"] == 1.0


def test_ask_endpoint_can_return_debug_trace(monkeypatch):
    from app.api import routes

    class FakeAnswer:
        def __init__(self):
            self.answer = "debug answer"
            self.sources = [
                SimpleNamespace(
                    source="report.pdf",
                    page=2,
                    chunk_index=7,
                    matched_child_chunk_index=1,
                    content_type="table",
                    table_index=0,
                    content="citation",
                )
            ]
            self.debug = {"query_variants": ["Q"], "final_chunks": []}

    monkeypatch.setattr(routes, "build_rag_service", lambda: type("Svc", (), {"answer": lambda self, **kwargs: FakeAnswer()})())

    client = TestClient(app)
    response = client.post(
        "/ask",
        json={
            "question": "Q",
            "top_k": 2,
            "debug": True,
            "rewrite_enabled": False,
            "hyde_enabled": True,
            "parent_hydration_enabled": False,
            "max_query_variants": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["query_variants"] == ["Q"]
    assert payload["sources"][0]["content_type"] == "table"
    assert payload["sources"][0]["matched_child_chunk_index"] == 1


def test_ask_endpoint_rejects_question_over_configured_length(monkeypatch):
    from app.api import routes

    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            max_question_chars=5,
            answer_cache_enabled=False,
            retrieval_top_k=4,
        ),
    )
    monkeypatch.setattr(routes, "build_rag_service", lambda: None)

    client = TestClient(app)
    response = client.post("/ask", json={"question": "123456", "top_k": 2})

    assert response.status_code == 422
    assert "max_question_chars" in response.json()["detail"]


def test_ask_endpoint_can_return_cached_answer(monkeypatch):
    from app.api import routes

    class FakeCache:
        def get(self, key):
            self.last_key = key
            return {
                "answer": "cached answer",
                "sources": [
                    {
                        "source": "doc.md",
                        "page": None,
                        "chunk_index": 0,
                        "matched_child_chunk_index": None,
                        "content_type": "text",
                        "table_index": None,
                        "content": "cached citation",
                    }
                ],
                "debug": {"cache": {"hit": True}},
            }

        def set(self, key, payload):
            raise AssertionError("cache should not be written on hit")

    def fail_build_service():
        raise AssertionError("RAG service should not be built on cache hit")

    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            max_question_chars=2000,
            answer_cache_enabled=True,
            answer_cache_ttl_seconds=300,
            answer_cache_backend="memory",
            redis_url="redis://localhost:6379/0",
            retrieval_top_k=4,
            chat_model="gpt-4o-mini",
            embedding_model="bge-m3",
            chroma_collection="test",
        ),
    )
    monkeypatch.setattr(routes, "build_answer_cache", lambda settings: FakeCache())
    monkeypatch.setattr(routes, "build_rag_service", fail_build_service)

    client = TestClient(app)
    response = client.post("/ask", json={"question": "Q", "top_k": 2, "debug": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "cached answer"
    assert payload["debug"]["cache"]["hit"] is True


def test_ask_endpoint_degrades_when_cache_read_fails(monkeypatch):
    from app.api import routes

    class BrokenCache:
        def get(self, key):
            raise RuntimeError("redis unavailable")

        def set(self, key, payload):
            self.payload = payload

    class FakeAnswer:
        answer = "fresh answer"
        sources = []
        debug = None

    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            max_question_chars=2000,
            answer_cache_enabled=True,
            answer_cache_ttl_seconds=300,
            answer_cache_backend="memory",
            redis_url="redis://localhost:6379/0",
            retrieval_top_k=4,
            chat_model="gpt-4o-mini",
            embedding_model="bge-m3",
            chroma_collection="test",
        ),
    )
    monkeypatch.setattr(routes, "build_answer_cache", lambda settings: BrokenCache())
    monkeypatch.setattr(routes, "build_rag_service", lambda: type("Svc", (), {"answer": lambda self, **kwargs: FakeAnswer()})())

    client = TestClient(app)
    response = client.post("/ask", json={"question": "Q", "top_k": 2})

    assert response.status_code == 200
    assert response.json()["answer"] == "fresh answer"


def test_corpus_status_endpoint_reports_index_files(tmp_path: Path, monkeypatch):
    from app.api import routes

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "guide.md").write_text("RAG guide", encoding="utf-8")
    (documents_dir / "notes.tmp").write_text("ignored", encoding="utf-8")

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    bm25_path = chroma_dir / "bm25.jsonl"
    bm25_path.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    parent_path = chroma_dir / "parents.jsonl"
    parent_path.write_text('{"id":"p1"}\n', encoding="utf-8")

    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            documents_dir=documents_dir,
            bm25_corpus_path=bm25_path,
            parent_corpus_path=parent_path,
            chroma_dir=chroma_dir,
            chroma_collection="test_collection",
            document_index_version="local-index-v1",
            versioned_indexing_enabled=False,
        ),
    )
    monkeypatch.setattr(
        routes,
        "chroma_collection_status",
        lambda persist_dir, collection_name: {
            "persist_dir": str(persist_dir),
            "collection_name": collection_name,
            "exists": True,
            "chunk_count": 2,
            "size_bytes": 128,
            "updated_at": None,
            "error": None,
        },
    )

    status = routes.build_corpus_status()
    json.dumps(status, ensure_ascii=False)
    assert isinstance(status["bm25_corpus"], dict)
    assert isinstance(status["parent_corpus"], dict)
    assert isinstance(status["chroma"], dict)

    client = TestClient(app)
    response = client.get("/corpus/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_count"] == 1
    assert payload["chunk_count"] == 2
    assert payload["parent_chunk_count"] == 1
    assert payload["bm25_ready"] is True
    assert payload["ready"] is True
    assert payload["readiness_reason"] == "ready"
    assert payload["chroma_collection_name"] == "test_collection"


def test_corpus_status_reports_unready_empty_versioned_index(tmp_path: Path, monkeypatch):
    from app.api import routes

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "guide.md").write_text("RAG guide", encoding="utf-8")
    index_root = tmp_path / "indexes"
    active_path = index_root / "active_version.txt"
    active_path.parent.mkdir(parents=True)
    active_path.write_text("index-v1\n", encoding="utf-8")
    chroma_dir = index_root / "index-v1" / "chroma"
    chroma_dir.mkdir(parents=True)

    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            documents_dir=documents_dir,
            chroma_collection="test_collection",
            chroma_dir=tmp_path / "legacy" / "chroma",
            bm25_corpus_path=tmp_path / "legacy" / "bm25.jsonl",
            parent_corpus_path=tmp_path / "legacy" / "parents.jsonl",
            index_root_dir=index_root,
            active_index_version_path=active_path,
            document_index_version="configured",
            versioned_indexing_enabled=True,
        ),
    )
    monkeypatch.setattr(
        routes,
        "chroma_collection_status",
        lambda persist_dir, collection_name: {
            "persist_dir": str(persist_dir),
            "collection_name": collection_name,
            "exists": True,
            "chunk_count": 0,
            "size_bytes": 0,
            "updated_at": None,
            "error": None,
        },
    )

    status = routes.build_corpus_status()

    assert status["active_index_version"] == "index-v1"
    assert status["index_dir"] == str(index_root / "index-v1")
    assert status["ready"] is False
    assert status["readiness_reason"] == "missing_chroma_chunks"
