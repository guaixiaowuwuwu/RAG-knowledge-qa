from types import SimpleNamespace
from pathlib import Path

from fastapi.testclient import TestClient

from app.integrations.wecom.crypto import WeComMessageCrypto, calculate_signature, parse_encrypted_xml
from app.integrations.wecom.schemas import WeComReply
from app.main import app


TOKEN = "test-token"
CORP_ID = "wxcorp123"
AES_KEY = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"


def _settings(**overrides):
    values = {
        "wecom_enabled": True,
        "wecom_corp_id": CORP_ID,
        "wecom_agent_id": "1000001",
        "wecom_secret": "secret",
        "wecom_token": TOKEN,
        "wecom_encoding_aes_key": AES_KEY,
        "wecom_callback_path": "/integrations/wecom/callback",
        "wecom_response_mode": "passive",
        "wecom_user_mapping_path": "unused.json",
        "wecom_api_base_url": "https://qyapi.weixin.qq.com/cgi-bin",
        "wecom_request_timeout_seconds": 1.0,
        "wecom_retry_count": 0,
        "retrieval_top_k": 4,
        "default_tenant_id": "default",
        "permission_version": "local-v1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_url_verification_rejects_invalid_signature(monkeypatch):
    from app.integrations.wecom import routes

    monkeypatch.setattr(routes, "get_settings", lambda: _settings())
    client = TestClient(app)

    response = client.get(
        "/integrations/wecom/callback",
        params={
            "msg_signature": "bad",
            "timestamp": "1700000000",
            "nonce": "n",
            "echostr": "abc",
        },
    )

    assert response.status_code == 403


def test_url_verification_accepts_valid_encrypted_echostr(monkeypatch):
    from app.integrations.wecom import routes

    monkeypatch.setattr(routes, "get_settings", lambda: _settings())
    crypto = WeComMessageCrypto(token=TOKEN, encoding_aes_key=AES_KEY, corp_id=CORP_ID)
    echostr = crypto.encrypt("verified")
    signature = calculate_signature(TOKEN, "1700000000", "n", echostr)
    client = TestClient(app)

    response = client.get(
        "/integrations/wecom/callback",
        params={
            "msg_signature": signature,
            "timestamp": "1700000000",
            "nonce": "n",
            "echostr": echostr,
        },
    )

    assert response.status_code == 200
    assert response.text == "verified"


def test_text_callback_calls_handler_with_wecom_user(monkeypatch):
    from app.integrations.wecom import routes

    captured = {}

    class FakeHandler:
        def handle_text_message(self, message):
            captured["message"] = message
            return SimpleNamespace(reply=WeComReply(content="RAG answer"))

    monkeypatch.setattr(routes, "get_settings", lambda: _settings(wecom_response_mode="passive"))
    monkeypatch.setattr(routes, "build_wecom_message_handler", lambda: FakeHandler())

    plaintext = Path("tests/fixtures/wecom/text_message.xml").read_text(encoding="utf-8")
    crypto = WeComMessageCrypto(token=TOKEN, encoding_aes_key=AES_KEY, corp_id=CORP_ID)
    encrypted = crypto.encrypt(plaintext)
    signature = calculate_signature(TOKEN, "1700000000", "n", encrypted)
    body = f"<xml><Encrypt><![CDATA[{encrypted}]]></Encrypt></xml>"
    client = TestClient(app)

    response = client.post(
        "/integrations/wecom/callback",
        params={
            "msg_signature": signature,
            "timestamp": "1700000000",
            "nonce": "n",
        },
        content=body,
    )

    assert response.status_code == 200
    assert captured["message"].from_user_name == "alice"
    assert captured["message"].content == "制度是什么？"
    encrypted_reply = parse_encrypted_xml(response.text)
    assert crypto.decrypt(encrypted_reply).find("RAG answer") >= 0


def test_active_mode_returns_success_after_handler(monkeypatch):
    from app.integrations.wecom import routes

    class FakeHandler:
        def handle_text_message(self, message):
            return SimpleNamespace(reply=WeComReply(content="sent actively"))

    monkeypatch.setattr(routes, "get_settings", lambda: _settings(wecom_response_mode="active"))
    monkeypatch.setattr(routes, "build_wecom_message_handler", lambda: FakeHandler())

    plaintext = Path("tests/fixtures/wecom/text_message.xml").read_text(encoding="utf-8")
    crypto = WeComMessageCrypto(token=TOKEN, encoding_aes_key=AES_KEY, corp_id=CORP_ID)
    encrypted = crypto.encrypt(plaintext)
    signature = calculate_signature(TOKEN, "1700000000", "n", encrypted)
    client = TestClient(app)

    response = client.post(
        "/integrations/wecom/callback",
        params={
            "msg_signature": signature,
            "timestamp": "1700000000",
            "nonce": "n",
        },
        content=f"<xml><Encrypt><![CDATA[{encrypted}]]></Encrypt></xml>",
    )

    assert response.status_code == 200
    assert response.text == "success"
