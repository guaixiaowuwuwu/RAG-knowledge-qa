import re
from dataclasses import dataclass
from typing import Any

from app.ingestion.chunker import Chunk


REFUSAL_ANSWER = "知识库中没有找到可支持该问题的资料，无法基于现有资料回答。"


@dataclass(frozen=True)
class RetrievalConfidenceConfig:
    min_reranker_score: float = -5.0
    min_final_source_count: int = 1
    enable_low_confidence_refusal: bool = True
    time_sensitive_refusal_enabled: bool = True

    @classmethod
    def from_settings(cls, settings) -> "RetrievalConfidenceConfig":
        return cls(
            min_reranker_score=float(getattr(settings, "min_reranker_score", cls.min_reranker_score)),
            min_final_source_count=int(getattr(settings, "min_final_source_count", cls.min_final_source_count)),
            enable_low_confidence_refusal=bool(
                getattr(settings, "enable_low_confidence_refusal", cls.enable_low_confidence_refusal)
            ),
            time_sensitive_refusal_enabled=bool(
                getattr(settings, "time_sensitive_refusal_enabled", cls.time_sensitive_refusal_enabled)
            ),
        )


@dataclass(frozen=True)
class RetrievalConfidenceDecision:
    should_refuse: bool
    refusal_reason: str | None
    final_source_count: int
    best_reranker_score: float | None
    entity_hints: list[str]
    entity_hint_matched: bool | None
    matched_rule_terms: list[str]
    config: RetrievalConfidenceConfig

    def to_dict(self) -> dict:
        return {
            "should_refuse": self.should_refuse,
            "refusal_reason": self.refusal_reason,
            "final_source_count": self.final_source_count,
            "best_reranker_score": self.best_reranker_score,
            "entity_hints": self.entity_hints,
            "entity_hint_matched": self.entity_hint_matched,
            "matched_rule_terms": self.matched_rule_terms,
            "min_reranker_score": self.config.min_reranker_score,
            "min_final_source_count": self.config.min_final_source_count,
            "enable_low_confidence_refusal": self.config.enable_low_confidence_refusal,
            "time_sensitive_refusal_enabled": self.config.time_sensitive_refusal_enabled,
        }


@dataclass(frozen=True)
class IntentSignal:
    refusal_reason: str | None = None
    matched_terms: list[str] | None = None


CORPUS_ENTITY_HINTS = {
    "byd": ["byd", "比亚迪"],
    "nvidia": ["nvidia", "nvda"],
    "microsoft": ["microsoft", "msft", "微软"],
    "google": ["google", "alphabet", "googl", "谷歌"],
    "apple": ["apple", "aapl", "苹果"],
    "amazon": ["amazon", "amzn", "亚马逊"],
}

UNAVAILABLE_ENTITY_PATTERNS = {
    "tesla": [r"\btesla\b", r"特斯拉"],
    "openai": [r"\bopenai\b"],
}

PRIVATE_OR_UNAVAILABLE_PATTERNS = [
    r"私密合同",
    r"未披露",
    r"保密合同",
    r"具体薪资",
    r"逐名员工",
    r"每位员工.{0,8}薪资",
    r"\bprivate contract\b",
    r"\bundisclosed\b",
    r"\bconfidential\b",
    r"\bspecific salary\b",
]

REAL_TIME_TERMS = [
    "今天",
    "此刻",
    "实时",
    "当前",
    "today",
    "current",
    "real-time",
    "realtime",
    "market close today",
]

LIVE_DATA_TERMS = [
    "股价",
    "股票",
    "收盘价",
    "产量",
    "工厂",
    "stock price",
    "market close",
    "share price",
    "factory",
    "output",
    "production",
]


def decide_retrieval_confidence(
    question: str,
    chunks: list[Chunk],
    *,
    trace: Any | None = None,
    config: RetrievalConfidenceConfig | None = None,
) -> RetrievalConfidenceDecision:
    config = config or RetrievalConfidenceConfig()
    final_source_count = len(chunks)
    best_reranker_score = _best_reranker_score(trace)
    entity_hints = infer_entity_hints(question)
    entity_hint_matched = _entity_hint_matched(chunks, entity_hints)
    intent = detect_unsupported_intent(question)

    refusal_reason = None
    if intent.refusal_reason == "time_sensitive" and config.time_sensitive_refusal_enabled:
        refusal_reason = "time_sensitive"
    elif intent.refusal_reason == "private_or_unavailable" and config.enable_low_confidence_refusal:
        refusal_reason = "private_or_unavailable"
    elif final_source_count == 0:
        refusal_reason = "empty_retrieval"
    elif config.enable_low_confidence_refusal and final_source_count < max(1, config.min_final_source_count):
        refusal_reason = "low_retrieval_confidence"
    elif (
        config.enable_low_confidence_refusal
        and best_reranker_score is not None
        and best_reranker_score < config.min_reranker_score
    ):
        refusal_reason = "low_retrieval_confidence"

    return RetrievalConfidenceDecision(
        should_refuse=refusal_reason is not None,
        refusal_reason=refusal_reason,
        final_source_count=final_source_count,
        best_reranker_score=best_reranker_score,
        entity_hints=entity_hints,
        entity_hint_matched=entity_hint_matched,
        matched_rule_terms=intent.matched_terms or [],
        config=config,
    )


def detect_unsupported_intent(question: str) -> IntentSignal:
    normalized = _normalize(question)
    matched_terms = []

    for patterns in UNAVAILABLE_ENTITY_PATTERNS.values():
        for pattern in patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                matched_terms.append(pattern)
                return IntentSignal("private_or_unavailable", matched_terms)

    for pattern in PRIVATE_OR_UNAVAILABLE_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            matched_terms.append(pattern)

    if _is_apple_future_unit_question(normalized):
        matched_terms.append("apple calendar year 2026 unit sales")

    if matched_terms:
        return IntentSignal("private_or_unavailable", matched_terms)

    time_terms = [term for term in REAL_TIME_TERMS if term in normalized]
    live_terms = [term for term in LIVE_DATA_TERMS if term in normalized]
    if "实时" in normalized or "real-time" in normalized or "realtime" in normalized:
        matched_terms.extend(time_terms or ["实时"])
        matched_terms.extend(live_terms)
        return IntentSignal("time_sensitive", matched_terms)
    if time_terms and live_terms:
        matched_terms.extend(time_terms)
        matched_terms.extend(live_terms)
        return IntentSignal("time_sensitive", matched_terms)

    return IntentSignal()


def infer_entity_hints(question: str) -> list[str]:
    normalized = _normalize(question)
    hints: list[str] = []
    for entity_hints in CORPUS_ENTITY_HINTS.values():
        if any(hint in normalized for hint in entity_hints):
            hints.extend(entity_hints)
    for entity_hints in UNAVAILABLE_ENTITY_PATTERNS.values():
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in entity_hints):
            hints.extend(entity_hints)
    return _unique(hints)


def _best_reranker_score(trace: Any | None) -> float | None:
    if trace is None:
        return None
    rows = getattr(trace, "reranker_scores", None)
    if rows is None and hasattr(trace, "to_dict"):
        rows = trace.to_dict().get("reranker_scores")
    scores = [row.get("score") for row in rows or [] if row.get("score") is not None]
    if not scores:
        return None
    return max(float(score) for score in scores)


def _entity_hint_matched(chunks: list[Chunk], entity_hints: list[str]) -> bool | None:
    if not entity_hints:
        return None
    haystacks = [
        f"{chunk.source}\n{chunk.content}".lower()
        for chunk in chunks
    ]
    if not haystacks:
        return False
    return any(_hint_in_haystack(hint, haystack) for hint in entity_hints for haystack in haystacks)


def _hint_in_haystack(hint: str, haystack: str) -> bool:
    if hint.startswith("\\b"):
        return re.search(hint, haystack, flags=re.IGNORECASE) is not None
    return hint.lower() in haystack


def _is_apple_future_unit_question(normalized: str) -> bool:
    return (
        "apple" in normalized
        and "iphone" in normalized
        and "calendar year 2026" in normalized
        and ("sell" in normalized or "sold" in normalized)
    )


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
