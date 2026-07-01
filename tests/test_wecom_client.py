import httpx

from app.integrations.wecom.client import WeComClient
from app.integrations.wecom.config import WeComSettings


def test_access_token_is_cached_and_reused_for_send_text():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/gettoken"):
            return httpx.Response(200, json={"errcode": 0, "access_token": "token-1", "expires_in": 7200})
        if request.url.path.endswith("/message/send"):
            assert request.url.params["access_token"] == "token-1"
            payload = request.read().decode("utf-8")
            assert '"touser":"alice"' in payload
            assert '"content":"hello"' in payload
            return httpx.Response(200, json={"errcode": 0, "errmsg": "ok"})
        raise AssertionError(f"unexpected request {request.url}")

    client = WeComClient(
        WeComSettings(
            corp_id="corp",
            secret="secret",
            agent_id="1000001",
            api_base_url="https://wecom.test/cgi-bin",
            retry_count=0,
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.get_access_token() == "token-1"
    assert client.send_text(touser="alice", content="hello")["errcode"] == 0
    assert [request.url.path for request in requests].count("/cgi-bin/gettoken") == 1


def test_get_user_info_by_code_uses_access_token_and_code():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/gettoken"):
            return httpx.Response(200, json={"errcode": 0, "access_token": "token-1", "expires_in": 7200})
        assert request.url.path.endswith("/auth/getuserinfo")
        assert request.url.params["access_token"] == "token-1"
        assert request.url.params["code"] == "code-1"
        return httpx.Response(200, json={"errcode": 0, "userid": "alice"})

    client = WeComClient(
        WeComSettings(
            corp_id="corp",
            secret="secret",
            api_base_url="https://wecom.test/cgi-bin",
            retry_count=0,
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.get_user_info_by_code("code-1")["userid"] == "alice"
