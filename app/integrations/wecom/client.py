import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.integrations.wecom.config import WeComSettings


logger = logging.getLogger(__name__)


class WeComClientError(RuntimeError):
    pass


@dataclass
class _CachedToken:
    value: str
    expires_at: float


class WeComClient:
    def __init__(
        self,
        settings: WeComSettings,
        *,
        http_client: httpx.Client | None = None,
    ):
        self.settings = settings
        self._client = http_client or httpx.Client(timeout=settings.request_timeout_seconds)
        self._owns_client = http_client is None
        self._cached_token: _CachedToken | None = None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def get_access_token(self) -> str:
        now = time.time()
        if self._cached_token and self._cached_token.expires_at > now + 60:
            return self._cached_token.value

        payload = self._request_json(
            "GET",
            "/gettoken",
            params={"corpid": self.settings.corp_id, "corpsecret": self.settings.secret},
            include_token=False,
        )
        token = str(payload.get("access_token") or "")
        if not token:
            raise WeComClientError("WeCom gettoken response did not include access_token.")
        expires_in = int(payload.get("expires_in") or 7200)
        self._cached_token = _CachedToken(value=token, expires_at=now + expires_in)
        return token

    def get_user_info_by_code(self, code: str) -> dict[str, Any]:
        return self._request_json(
            "GET",
            "/auth/getuserinfo",
            params={"code": code},
        )

    def get_user(self, userid: str) -> dict[str, Any]:
        return self._request_json(
            "GET",
            "/user/get",
            params={"userid": userid},
        )

    def send_text(self, *, touser: str, content: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/message/send",
            json={
                "touser": touser,
                "msgtype": "text",
                "agentid": int(self.settings.agent_id) if str(self.settings.agent_id).isdigit() else self.settings.agent_id,
                "text": {"content": content},
                "safe": 0,
            },
        )

    def send_textcard(self, *, touser: str, title: str, description: str, url: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/message/send",
            json={
                "touser": touser,
                "msgtype": "textcard",
                "agentid": int(self.settings.agent_id) if str(self.settings.agent_id).isdigit() else self.settings.agent_id,
                "textcard": {
                    "title": title,
                    "description": description,
                    "url": url,
                },
                "safe": 0,
            },
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        include_token: bool = True,
    ) -> dict[str, Any]:
        params = dict(params or {})
        if include_token:
            params["access_token"] = self.get_access_token()

        last_error: Exception | None = None
        for attempt in range(self.settings.retry_count + 1):
            try:
                response = self._client.request(
                    method,
                    f"{self.settings.api_base_url}{path}",
                    params=params,
                    json=json,
                    timeout=self.settings.request_timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                errcode = int(payload.get("errcode", 0) or 0)
                if errcode != 0:
                    raise WeComClientError(f"WeCom API error errcode={errcode} errmsg={payload.get('errmsg', '')}")
                return payload
            except (httpx.HTTPError, ValueError, WeComClientError) as exc:
                last_error = exc
                if attempt >= self.settings.retry_count:
                    break
                time.sleep(0.1 * (attempt + 1))
        raise WeComClientError(f"WeCom API request failed: {last_error}") from last_error
