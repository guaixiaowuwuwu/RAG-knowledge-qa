import hashlib
import json
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Protocol


logger = logging.getLogger(__name__)


class AnswerCache(Protocol):
    def get(self, key: str) -> dict | None:
        ...

    def set(self, key: str, payload: dict) -> None:
        ...


class NullAnswerCache:
    def get(self, key: str) -> dict | None:
        return None

    def set(self, key: str, payload: dict) -> None:
        return None


class InMemoryAnswerCache:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._values: dict[str, tuple[float, dict]] = {}

    def get(self, key: str) -> dict | None:
        row = self._values.get(key)
        if row is None:
            return None

        expires_at, payload = row
        if expires_at <= time.time():
            self._values.pop(key, None)
            return None
        return payload

    def set(self, key: str, payload: dict) -> None:
        self._values[key] = (time.time() + self.ttl_seconds, payload)


class RedisAnswerCache:
    def __init__(self, url: str, ttl_seconds: int):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Redis cache requires the optional `redis` package.") from exc

        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> dict | None:
        value = self.client.get(key)
        if value is None:
            return None
        return json.loads(value)

    def set(self, key: str, payload: dict) -> None:
        self.client.setex(key, self.ttl_seconds, json.dumps(payload, ensure_ascii=False))


def build_answer_cache(settings) -> AnswerCache:
    if not getattr(settings, "answer_cache_enabled", False):
        return NullAnswerCache()

    ttl_seconds = int(getattr(settings, "answer_cache_ttl_seconds", 300))
    backend = str(getattr(settings, "answer_cache_backend", "redis")).lower()

    if backend == "memory":
        return InMemoryAnswerCache(ttl_seconds=ttl_seconds)

    if backend == "redis":
        try:
            return RedisAnswerCache(url=str(getattr(settings, "redis_url")), ttl_seconds=ttl_seconds)
        except Exception as exc:
            logger.warning("answer_cache_unavailable backend=redis error=%s", exc)
            return NullAnswerCache()

    logger.warning("answer_cache_unknown_backend backend=%s", backend)
    return NullAnswerCache()


def build_cache_key(question: str, top_k: int, settings, retrieval_options: object | None = None) -> str:
    payload = {
        "question": question.strip(),
        "top_k": top_k,
        "chat_model": getattr(settings, "chat_model", ""),
        "embedding_model": getattr(settings, "embedding_model", ""),
        "chroma_collection": getattr(settings, "chroma_collection", ""),
        "retrieval_options": normalize_options(retrieval_options),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"rag-answer:{digest}"


def normalize_options(options: object | None) -> dict:
    if options is None:
        return {}
    if isinstance(options, dict):
        return options
    if is_dataclass(options):
        return asdict(options)
    return {
        key: value
        for key, value in vars(options).items()
        if not key.startswith("_")
    }
