from fastapi.testclient import TestClient

from app.main import app


def test_root_serves_frontend_html():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "RAG Knowledge QA" in response.text
    assert "/static/styles.css" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_are_served():
    client = TestClient(app)

    css_response = client.get("/static/styles.css")
    js_response = client.get("/static/app.js")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
