from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


LabelSet = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MetricSample:
    name: str
    value: float
    labels: dict[str, str]


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, LabelSet], float] = defaultdict(float)
        self._histograms: dict[tuple[str, LabelSet], list[float]] = defaultdict(list)

    def increment(self, name: str, amount: float = 1.0, **labels: object) -> None:
        with self._lock:
            self._counters[(name, _labels(labels))] += float(amount)

    def observe(self, name: str, value: float, **labels: object) -> None:
        with self._lock:
            self._histograms[(name, _labels(labels))].append(float(value))

    def samples(self) -> list[MetricSample]:
        with self._lock:
            samples = [
                MetricSample(name=name, value=value, labels=dict(labels))
                for (name, labels), value in self._counters.items()
            ]
            for (name, labels), values in self._histograms.items():
                label_dict = dict(labels)
                samples.append(MetricSample(name=f"{name}_count", value=float(len(values)), labels=label_dict))
                samples.append(MetricSample(name=f"{name}_sum", value=sum(values), labels=label_dict))
        return sorted(samples, key=lambda sample: (sample.name, sorted(sample.labels.items())))

    def render_prometheus(self) -> str:
        lines = []
        for sample in self.samples():
            lines.append(f"{sample.name}{_render_labels(sample.labels)} {_format_value(sample.value)}")
        return "\n".join(lines) + ("\n" if lines else "")

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


metrics = MetricsRegistry()


def record_ingestion_job_status(status: str) -> None:
    metrics.increment("rag_ingestion_jobs_total", status=status)


def _labels(labels: dict[str, object]) -> LabelSet:
    return tuple(sorted((str(key), str(value)) for key, value in labels.items() if value is not None))


def _render_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{key}="{_escape_label(value)}"' for key, value in sorted(labels.items()))
    return "{" + rendered + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_value(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(round(value, 6))
