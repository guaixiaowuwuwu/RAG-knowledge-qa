import json
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "question",
    "ground_truth",
    "expected_sources",
    "expected_answer_keywords",
}


class EvalDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class EvalCase:
    question: str
    ground_truth: str
    expected_sources: list[str]
    expected_answer_keywords: list[str] = field(default_factory=list)
    expected_pages: dict[str, list[int]] = field(default_factory=dict)
    expected_chunk_keywords: list[str] = field(default_factory=list)
    evidence_notes: str = ""
    id: str = ""
    category: str = ""
    difficulty: str = ""
    language: str = ""
    notes: str = ""
    is_negative: bool = False


def load_eval_cases(
    path: Path,
    *,
    check_source_files: bool = False,
    source_base_dir: Path | None = None,
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(parse_eval_case(row, line_number=line_number))

    errors = validate_eval_cases(
        cases,
        check_source_files=check_source_files,
        source_base_dir=source_base_dir or Path.cwd(),
    )
    if errors:
        raise EvalDatasetError("\n".join(errors))
    return cases


def parse_eval_case(row: dict, *, line_number: int) -> EvalCase:
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        raise EvalDatasetError(f"line {line_number}: missing required fields: {', '.join(missing)}")

    expected_sources = _string_list(row["expected_sources"], "expected_sources", line_number)
    expected_answer_keywords = _string_list(
        row["expected_answer_keywords"],
        "expected_answer_keywords",
        line_number,
    )
    is_negative = bool(row.get("is_negative", False))
    expected_pages = _page_map(row.get("expected_pages", {}), expected_sources, line_number)

    return EvalCase(
        id=str(row["id"]).strip(),
        question=str(row["question"]).strip(),
        ground_truth=str(row["ground_truth"]).strip(),
        expected_sources=expected_sources,
        expected_answer_keywords=expected_answer_keywords,
        expected_pages=expected_pages,
        expected_chunk_keywords=_string_list(row.get("expected_chunk_keywords", []), "expected_chunk_keywords", line_number),
        evidence_notes=str(row.get("evidence_notes", "")).strip(),
        category=str(row.get("category", "")).strip(),
        difficulty=str(row.get("difficulty", "")).strip(),
        language=str(row.get("language", "")).strip(),
        notes=str(row.get("notes", "")).strip(),
        is_negative=is_negative,
    )


def validate_eval_cases(
    cases: list[EvalCase],
    *,
    check_source_files: bool = False,
    source_base_dir: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    source_base_dir = source_base_dir or Path.cwd()

    for index, case in enumerate(cases, start=1):
        label = case.id or f"line {index}"
        if not case.id:
            errors.append(f"{label}: id must not be empty")
        elif case.id in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(case.id)

        if not case.question:
            errors.append(f"{label}: question must not be empty")
        if not case.ground_truth:
            errors.append(f"{label}: ground_truth must not be empty")
        if not case.is_negative and not case.expected_sources:
            errors.append(f"{label}: expected_sources must not be empty unless is_negative is true")
        if not case.is_negative and not case.expected_answer_keywords:
            errors.append(f"{label}: expected_answer_keywords must not be empty for positive cases")

        if check_source_files:
            for source in case.expected_sources:
                if not _source_exists(source, source_base_dir):
                    errors.append(f"{label}: expected source does not exist: {source}")
        for source, pages in case.expected_pages.items():
            if source not in case.expected_sources:
                errors.append(f"{label}: expected_pages source is not in expected_sources: {source}")
            if any(page < 1 for page in pages):
                errors.append(f"{label}: expected_pages must contain positive page numbers")

    return errors


def _string_list(value, field_name: str, line_number: int) -> list[str]:
    if not isinstance(value, list):
        raise EvalDatasetError(f"line {line_number}: {field_name} must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def _page_map(value, expected_sources: list[str], line_number: int) -> dict[str, list[int]]:
    if value in ({}, [], None):
        return {}

    if isinstance(value, list):
        if len(expected_sources) != 1:
            raise EvalDatasetError(f"line {line_number}: expected_pages list form requires exactly one expected source")
        return {expected_sources[0]: _int_list(value, "expected_pages", line_number)}

    if not isinstance(value, dict):
        raise EvalDatasetError(f"line {line_number}: expected_pages must be an object or list")

    pages_by_source: dict[str, list[int]] = {}
    for source, pages in value.items():
        source_key = str(source).strip()
        if not source_key:
            raise EvalDatasetError(f"line {line_number}: expected_pages source must not be empty")
        pages_by_source[source_key] = _int_list(pages, "expected_pages", line_number)
    return pages_by_source


def _int_list(value, field_name: str, line_number: int) -> list[int]:
    if not isinstance(value, list):
        raise EvalDatasetError(f"line {line_number}: {field_name} values must be lists")
    pages = []
    for item in value:
        try:
            pages.append(int(item))
        except (TypeError, ValueError) as exc:
            raise EvalDatasetError(f"line {line_number}: {field_name} values must be integers") from exc
    return pages


def _source_exists(source: str, source_base_dir: Path) -> bool:
    source_path = Path(source)
    if source_path.is_absolute():
        return source_path.exists()
    return (source_base_dir / source_path).exists()
