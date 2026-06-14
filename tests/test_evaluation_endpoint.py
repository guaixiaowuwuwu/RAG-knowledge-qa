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
