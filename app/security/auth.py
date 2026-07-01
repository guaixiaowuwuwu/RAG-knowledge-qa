from secrets import compare_digest
from typing import Iterable

from fastapi import Depends, HTTPException, Request, status

from app.core.config import get_settings
from app.security.context import RequestContext


ADMIN_ROLE = "admin"


def get_request_context(request: Request) -> RequestContext:
    state_context = getattr(request.state, "request_context", None)
    if isinstance(state_context, RequestContext):
        return state_context

    settings = get_settings()
    tenant_id = str(getattr(settings, "default_tenant_id", "default"))
    permission_version = str(getattr(settings, "permission_version", "local-v1"))

    if not _truthy(getattr(settings, "auth_enabled", False)):
        context = RequestContext.local_dev(
            tenant_id=tenant_id,
            permission_version=permission_version,
        )
        request.state.request_context = context
        return context

    api_key = _extract_api_key(request)
    if api_key:
        if _matches_any(api_key, _split_csv(getattr(settings, "admin_api_keys", ""))):
            context = RequestContext.admin_api_key(
                tenant_id=tenant_id,
                permission_version=permission_version,
            )
            request.state.request_context = context
            return context
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_authenticated(request: Request, context: RequestContext = Depends(get_request_context)) -> RequestContext:
    if not context.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.request_context = context
    return context


def require_admin(request: Request, context: RequestContext = Depends(get_request_context)) -> RequestContext:
    if not context.has_role(ADMIN_ROLE):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    request.state.request_context = context
    return context


def _extract_api_key(request: Request) -> str | None:
    for header_name in ("x-admin-api-key", "x-api-key"):
        value = request.headers.get(header_name)
        if value:
            return value.strip()

    authorization = request.headers.get("authorization")
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return None


def _matches_any(candidate: str, allowed_values: Iterable[str]) -> bool:
    return any(compare_digest(candidate, allowed) for allowed in allowed_values)


def _split_csv(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_values = value.split(",")
    elif isinstance(value, Iterable):
        raw_values = [str(item) for item in value]
    else:
        raw_values = [str(value)]
    return tuple(item.strip() for item in raw_values if item and item.strip())


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
