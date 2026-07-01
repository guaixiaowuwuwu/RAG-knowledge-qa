from dataclasses import dataclass
from pathlib import Path
from typing import Literal


WeComResponseMode = Literal["active", "passive"]


@dataclass(frozen=True)
class WeComSettings:
    enabled: bool = False
    corp_id: str = ""
    agent_id: str = ""
    secret: str = ""
    token: str = ""
    encoding_aes_key: str = ""
    callback_path: str = "/integrations/wecom/callback"
    response_mode: WeComResponseMode = "active"
    user_mapping_path: Path = Path("data/runtime/wecom_users.json")
    api_base_url: str = "https://qyapi.weixin.qq.com/cgi-bin"
    request_timeout_seconds: float = 8.0
    retry_count: int = 2

    @property
    def encrypted_callbacks_enabled(self) -> bool:
        return bool(self.encoding_aes_key)


def wecom_settings_from_app_settings(settings: object) -> WeComSettings:
    return WeComSettings(
        enabled=bool(getattr(settings, "wecom_enabled", False)),
        corp_id=str(getattr(settings, "wecom_corp_id", "") or ""),
        agent_id=str(getattr(settings, "wecom_agent_id", "") or ""),
        secret=str(getattr(settings, "wecom_secret", "") or ""),
        token=str(getattr(settings, "wecom_token", "") or ""),
        encoding_aes_key=str(getattr(settings, "wecom_encoding_aes_key", "") or ""),
        callback_path=str(getattr(settings, "wecom_callback_path", "/integrations/wecom/callback") or "/integrations/wecom/callback"),
        response_mode=_normalize_response_mode(getattr(settings, "wecom_response_mode", "active")),
        user_mapping_path=Path(getattr(settings, "wecom_user_mapping_path", Path("data/runtime/wecom_users.json"))),
        api_base_url=str(getattr(settings, "wecom_api_base_url", "https://qyapi.weixin.qq.com/cgi-bin") or "https://qyapi.weixin.qq.com/cgi-bin").rstrip("/"),
        request_timeout_seconds=float(getattr(settings, "wecom_request_timeout_seconds", 8.0)),
        retry_count=int(getattr(settings, "wecom_retry_count", 2)),
    )


def _normalize_response_mode(value: object) -> WeComResponseMode:
    normalized = str(value or "active").strip().lower()
    if normalized == "passive":
        return "passive"
    return "active"
