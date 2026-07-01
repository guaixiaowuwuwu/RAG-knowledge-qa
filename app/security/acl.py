import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from app.security.context import RequestContext


@dataclass(frozen=True)
class DocumentACL:
    tenant_id: str
    allowed_user_ids: tuple[str, ...] = field(default_factory=tuple)
    allowed_department_ids: tuple[str, ...] = field(default_factory=tuple)
    allowed_roles: tuple[str, ...] = field(default_factory=tuple)
    is_public: bool = False

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object], *, default_tenant_id: str = "default") -> "DocumentACL":
        return cls(
            tenant_id=str(metadata.get("tenant_id") or default_tenant_id),
            allowed_user_ids=_normalize_values(metadata.get("allowed_user_ids")),
            allowed_department_ids=_normalize_values(metadata.get("allowed_department_ids")),
            allowed_roles=_normalize_values(metadata.get("allowed_roles")),
            is_public=_normalize_bool(metadata.get("is_public")),
        )


@dataclass(frozen=True)
class RetrievalAccessFilter:
    tenant_id: str
    user_id: str
    department_ids: tuple[str, ...] = field(default_factory=tuple)
    roles: tuple[str, ...] = field(default_factory=tuple)
    permission_version: str = "local-v1"
    allow_missing_acl: bool = False

    @classmethod
    def from_context(cls, context: RequestContext) -> "RetrievalAccessFilter":
        return cls(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            department_ids=context.department_ids,
            roles=context.roles,
            permission_version=context.permission_version,
            allow_missing_acl=context.source == "local",
        )

    def can_access_metadata(self, metadata: Mapping[str, object]) -> bool:
        if self.allow_missing_acl and not _has_acl_metadata(metadata):
            return True
        acl = DocumentACL.from_metadata(metadata, default_tenant_id=self.tenant_id)
        context = RequestContext(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            department_ids=self.department_ids,
            roles=self.roles,
            permission_version=self.permission_version,
            source="local" if self.allow_missing_acl else "api_key",
        )
        return can_access_document(context, acl)

    def summary(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "user_hash": _hash_identifier(self.user_id),
            "department_count": len(self.department_ids),
            "role_count": len(self.roles),
            "permission_version": self.permission_version,
            "allow_missing_acl": self.allow_missing_acl,
        }


def can_access_document(context: RequestContext, acl: DocumentACL) -> bool:
    if context.tenant_id != acl.tenant_id:
        return False
    if acl.is_public:
        return True
    if context.user_id and context.user_id in acl.allowed_user_ids:
        return True
    if set(context.department_ids).intersection(acl.allowed_department_ids):
        return True
    if set(context.roles).intersection(acl.allowed_roles):
        return True
    return False


def _normalize_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                items = [str(item) for item in parsed]
            else:
                items = stripped.split(",")
        else:
            items = stripped.split(",")
    elif isinstance(value, Iterable):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    return tuple(item.strip() for item in items if item and item.strip())


def _normalize_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _has_acl_metadata(metadata: Mapping[str, object]) -> bool:
    return any(
        key in metadata
        for key in (
            "tenant_id",
            "allowed_user_ids",
            "allowed_department_ids",
            "allowed_roles",
            "is_public",
        )
    )


def _hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
