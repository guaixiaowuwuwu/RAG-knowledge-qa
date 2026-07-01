import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any


SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "x-admin-api-key",
    "api_key",
    "admin_api_key",
    "wecom_secret",
    "wecom_token",
    "wecom_encoding_aes_key",
    "encoding_aes_key",
    "encrypt",
    "encrypted_payload",
    "raw_document",
    "document",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "structured", None)
        if isinstance(extra, dict):
            payload.update(redact_mapping(extra))
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def log_json(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    structured = redact_mapping({"event": event, **fields})
    logger.log(
        level,
        json.dumps(structured, ensure_ascii=False, sort_keys=True),
        extra={"structured": structured},
    )


def hash_identifier(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def redact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _redact_value(key, value) for key, value in values.items()}


def _redact_value(key: str, value: Any) -> Any:
    normalized = key.lower().replace("-", "_")
    if normalized in SENSITIVE_KEYS or any(token in normalized for token in ("secret", "token", "api_key")):
        return "[REDACTED]"
    if isinstance(value, dict):
        return redact_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_redact_value(key, item) for item in value]
    return value
