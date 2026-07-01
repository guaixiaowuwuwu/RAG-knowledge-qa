import json
from pathlib import Path
from typing import Protocol

from app.integrations.wecom.schemas import WeComUserMapping


class WeComUserMappingStore(Protocol):
    def get_by_wecom_userid(self, wecom_userid: str) -> WeComUserMapping | None:
        ...


class JsonWeComUserMappingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def get_by_wecom_userid(self, wecom_userid: str) -> WeComUserMapping | None:
        for item in self._load_items():
            if str(item.get("wecom_userid", "")) == wecom_userid:
                return WeComUserMapping(
                    tenant_id=str(item.get("tenant_id") or "default"),
                    wecom_userid=wecom_userid,
                    system_user_id=str(item.get("system_user_id") or wecom_userid),
                    display_name=item.get("display_name"),
                    department_ids=tuple(str(value) for value in item.get("department_ids", []) if value),
                    roles=tuple(str(value) for value in item.get("roles", []) if value),
                    permission_version=str(item.get("permission_version") or "local-v1"),
                )
        return None

    def _load_items(self) -> list[dict]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("users", [])
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]
