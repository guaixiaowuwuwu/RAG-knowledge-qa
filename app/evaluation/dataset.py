import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    question: str
    ground_truth: str
    expected_sources: list[str]
    expected_answer_keywords: list[str] = field(default_factory=list)


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            EvalCase(
                question=str(row["question"]),
                ground_truth=str(row.get("ground_truth", "")),
                expected_sources=[str(source) for source in row.get("expected_sources", [])],
                expected_answer_keywords=[str(keyword) for keyword in row.get("expected_answer_keywords", [])],
            )
        )
    return cases
