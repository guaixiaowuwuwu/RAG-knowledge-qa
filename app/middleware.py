import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request

from app.core.config import get_settings
from app.ingestion.index_versions import get_active_index_version
from app.observability.logging import hash_identifier, log_json
from app.observability.metrics import metrics


logger = logging.getLogger("app.requests")


def setup_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid4().hex
        request.state.request_id = request_id
        started_at = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
            path = request.url.path
            metrics.increment(
                "rag_http_requests_total",
                method=request.method,
                path=path,
                status=str(status_code),
            )
            metrics.observe(
                "rag_http_request_duration_ms",
                duration_ms,
                method=request.method,
                path=path,
                status=str(status_code),
            )
            if status_code >= 500:
                metrics.increment(
                    "rag_http_errors_total",
                    method=request.method,
                    path=path,
                    status=str(status_code),
                )
            context = getattr(request.state, "request_context", None)
            log_json(
                logger,
                logging.INFO,
                "http_request",
                request_id=request_id,
                method=request.method,
                path=path,
                status=status_code,
                duration_ms=duration_ms,
                tenant_id=getattr(context, "tenant_id", None),
                user_id_hash=hash_identifier(getattr(context, "user_id", None)),
                index_version=_active_index_version(),
            )
            if "response" in locals():
                response.headers["x-request-id"] = request_id
                response.headers["x-process-time-ms"] = str(duration_ms)


def _active_index_version() -> str | None:
    try:
        return get_active_index_version(get_settings())
    except Exception:
        return None
