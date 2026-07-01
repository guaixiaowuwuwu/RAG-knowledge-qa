from dataclasses import dataclass, field
from typing import Literal


ContextSource = Literal["local", "api_key", "wecom"]


@dataclass(frozen=True)
class RequestContext:
    tenant_id: str
    user_id: str
    display_name: str | None = None
    department_ids: tuple[str, ...] = field(default_factory=tuple)
    roles: tuple[str, ...] = field(default_factory=tuple)
    permission_version: str = "local-v1"
    source: ContextSource = "local"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", str(self.tenant_id))
        object.__setattr__(self, "user_id", str(self.user_id))
        object.__setattr__(self, "department_ids", _normalize_tuple(self.department_ids))
        object.__setattr__(self, "roles", _normalize_tuple(self.roles))
        object.__setattr__(self, "permission_version", str(self.permission_version))

    @classmethod
    def local_dev(cls, *, tenant_id: str, permission_version: str) -> "RequestContext":
        return cls(
            tenant_id=tenant_id,
            user_id="local-dev",
            display_name="Local Dev",
            department_ids=(),
            roles=("admin",),
            permission_version=permission_version,
            source="local",
        )

    @classmethod
    def admin_api_key(cls, *, tenant_id: str, permission_version: str) -> "RequestContext":
        return cls(
            tenant_id=tenant_id,
            user_id="admin-api-key",
            display_name="Admin API Key",
            department_ids=(),
            roles=("admin",),
            permission_version=permission_version,
            source="api_key",
        )

    def has_role(self, role: str) -> bool:
        return role in self.roles


def _normalize_tuple(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)
