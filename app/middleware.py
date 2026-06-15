import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request


logger = logging.getLogger("app.requests")


def setup_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid4().hex
        started_at = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
            logger.info(
                "http_request request_id=%s method=%s path=%s status_code=%s duration_ms=%.3f",
                request_id,
                request.method,
                request.url.path,
                status_code,
                duration_ms,
            )
            if "response" in locals():
                response.headers["x-request-id"] = request_id
                response.headers["x-process-time-ms"] = str(duration_ms)
