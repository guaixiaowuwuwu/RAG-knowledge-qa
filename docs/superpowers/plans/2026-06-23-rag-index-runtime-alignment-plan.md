# RAG 索引运行态与评估对齐执行计划

> **For Codex / agentic workers:** 这份计划用于修复当前项目的最高优先级阻塞：API 实际运行索引为空、评估脚本与 API 索引路径不一致、最新答案评估为 0 分。按任务顺序执行；先补测试，再改代码，再重建索引和重新评估。不要在修复前继续扩展企业微信或生产功能。

**Goal:** 让本地运行、API、评估脚本、答案评估和文档都使用同一套索引路径与 active index 语义，恢复可演示的完整问答能力，并生成可信的最新评估报告。

**Current observed state, 2026-06-23:**

- Full Python tests pass: `165 passed`.
- JS tests pass: `14 passed`.
- Enterprise focused tests pass: `48 passed`.
- Current API runtime uses `get_index_paths(settings)`.
- Current active/default version is `local-index-v1`.
- `data/indexes/local-index-v1` exists but has empty Chroma collection and no `bm25_corpus.jsonl` / `parent_corpus.jsonl`.
- Legacy usable artifacts still exist under `data/chroma/`.
- Latest retrieval report looks good, but may use legacy paths.
- Latest answer report is invalid as a quality proof: `pass_rate=0.0`, `unknown_answer_rate=1.0`.

**Critical issue:** `versioned_indexing_enabled(settings)` currently returns true whenever settings has `index_root_dir` or `active_index_version_path`. Since `Settings` always has these fields, versioned indexing is always enabled, even in local sync mode.

---

## Target Behavior

- Local default mode should keep working with existing `data/chroma` artifacts unless versioned indexing is explicitly enabled.
- Versioned indexing should be opt-in for enterprise/async mode.
- API, `/corpus/status`, `scripts.ingest`, `scripts.evaluate`, and `scripts.evaluate_answers` must resolve index paths through one shared helper.
- `build_corpus_status()` must report non-empty chunks after a successful local index.
- Retrieval and answer evaluation reports must reflect the same index used by `/ask`.

---

## Phase A: Add Explicit Versioned Indexing Setting

**Problem:** Versioned indexing is implicitly always enabled because the setting fields exist.

**Decision:** Add `VERSIONED_INDEXING_ENABLED=false` as the explicit switch. Local dev default remains legacy `data/chroma`. Enterprise async/staging can set it to `true`.

### Files likely touched

- `app/core/config.py`
- `.env.example`
- `app/ingestion/index_versions.py`
- `tests/test_index_versions.py`
- `tests/test_phase5_config.py`

### Tasks

- [ ] Add setting:
  - `versioned_indexing_enabled: bool = Field(default=False, alias="VERSIONED_INDEXING_ENABLED")`

- [ ] Update `.env.example`:
  - `VERSIONED_INDEXING_ENABLED=false`
  - Keep `INDEX_ROOT_DIR` and `ACTIVE_INDEX_VERSION_PATH`, but document that they are used only when versioning is enabled.

- [ ] Change `versioned_indexing_enabled(settings)`.
  - Return `bool(getattr(settings, "versioned_indexing_enabled", False))`.
  - Do not infer enabled from `hasattr`.

- [ ] Keep `get_active_index_version(settings)` behavior.
  - If versioning disabled, it may return `DOCUMENT_INDEX_VERSION`, but `get_index_paths()` must still use legacy paths.

- [ ] Add/adjust tests.
  - Default `Settings()` has `versioned_indexing_enabled is False`.
  - With default settings, `get_index_paths(settings).chroma_dir == settings.chroma_dir`.
  - With `versioned_indexing_enabled=True`, `get_index_paths(settings).chroma_dir == settings.index_root_dir / version / "chroma"`.
  - Existing activation/listing tests still pass when versioning is enabled explicitly.

### Verification

```bash
.venv/bin/pytest tests/test_index_versions.py tests/test_phase5_config.py -v
```

### Acceptance

- Local default no longer points API at empty `data/indexes/local-index-v1`.
- Versioned indexing remains available when `VERSIONED_INDEXING_ENABLED=true`.

---

## Phase B: Align Evaluation Scripts With Runtime Index Paths

**Problem:** `scripts/evaluate.py` variant builders still read `settings.bm25_corpus_path` and `settings.parent_corpus_path`, while API uses `get_index_paths(settings)`.

**Decision:** All retriever construction paths must call the same runtime index resolver.

### Files likely touched

- `scripts/evaluate.py`
- `scripts/evaluate_answers.py`
- `app/api/routes.py`
- `tests/test_evaluation.py`
- `tests/test_api_retriever_factory.py`

### Tasks

- [ ] Update `scripts/evaluate.py`.
  - Import `get_index_paths`.
  - In `build_retriever_variant()`, resolve:
    - `index_paths = get_index_paths(settings)`
    - `BM25Retriever.from_jsonl(index_paths.bm25_corpus_path)`
    - `JsonlParentStore(index_paths.parent_corpus_path)`
  - Keep dense vector store via existing `build_vector_store()` because it already uses runtime paths.

- [ ] Update `config_snapshot(settings)`.
  - Include effective index paths:
    - `effective_index_version`
    - `effective_chroma_dir`
    - `effective_bm25_corpus_path`
    - `effective_parent_corpus_path`
    - `versioned_indexing_enabled`
  - Keep legacy config fields for backward comparison, but label effective fields clearly.

- [ ] Confirm `scripts/evaluate_answers.py`.
  - It builds service through `build_rag_service()`, so runtime path should already be aligned.
  - Add a test or assertion that answer eval report config includes effective index path from `config_snapshot`.

- [ ] Add tests.
  - Monkeypatch settings with distinct legacy and versioned paths.
  - Enable versioned indexing.
  - Assert `build_retriever_variant("hybrid-rerank-parent")` loads BM25 and parent corpus from `get_index_paths(settings)`, not legacy paths.
  - Assert report config contains effective path fields.

### Verification

```bash
.venv/bin/pytest tests/test_evaluation.py tests/test_api_retriever_factory.py tests/test_answer_evaluation.py -v
```

### Acceptance

- Retrieval reports and API use the same index artifacts.
- Future reports cannot silently use stale `data/chroma` when API uses `data/indexes`.

---

## Phase C: Fix Corpus Status And Readiness Semantics

**Problem:** Current `/corpus/status` can report an empty active index after versioned path switch. There is no explicit readiness status for "index exists but unusable".

**Decision:** Add a clear readiness field without changing existing fields.

### Files likely touched

- `app/api/routes.py`
- `app/api/schemas.py`
- `tests/test_evaluation_endpoint.py`
- `docs/production-readiness.md`

### Tasks

- [ ] Extend `CorpusStatusResponse`.
  - Add `active_index_version: str | None = None` if not already present.
  - Add `index_dir: str | None = None` if not already present.
  - Add `ready: bool`.
  - Add `readiness_reason: str | None`.

- [ ] Define readiness logic in `build_corpus_status()`.
  - Ready only if:
    - Chroma chunk count > 0.
    - BM25 corpus exists and count > 0.
    - Parent corpus exists and count > 0.
  - Reasons:
    - `missing_chroma_chunks`
    - `missing_bm25_corpus`
    - `missing_parent_corpus`
    - `ready`

- [ ] Update frontend only if it currently assumes old fields.
  - Show "待索引" when `ready=false`.
  - Do not block old display cards.

- [ ] Add tests.
  - Empty versioned index returns `ready=false`.
  - Complete legacy index returns `ready=true` when versioning disabled.
  - Response model validates through FastAPI.

### Verification

```bash
.venv/bin/pytest tests/test_evaluation_endpoint.py tests/test_web_static.py -v
node --test tests/js/ui-utils.test.mjs
```

### Acceptance

- `/corpus/status` tells the truth about whether `/ask` can retrieve.
- Empty active index cannot be mistaken for production-ready state.

---

## Phase D: Restore A Usable Local Index

**Problem:** Current active runtime index is empty. After Phase A, local default should use existing `data/chroma`; still, a clean rebuild must be validated.

**Decision:** Rebuild local legacy index first. Do not enable versioned indexing until legacy path is healthy. Then optionally run one versioned build smoke.

### Files likely touched

- No code files expected after earlier phases.
- Generated artifacts:
  - `data/chroma/`
  - `reports/`
  - Optional `data/indexes/<version>/`

### Tasks

- [ ] Confirm local default path after Phase A.

```bash
.venv/bin/python - <<'PY'
from app.core.config import Settings
from app.ingestion.index_versions import get_index_paths
s = Settings()
p = get_index_paths(s)
print("versioned:", s.versioned_indexing_enabled)
print("chroma:", p.chroma_dir)
print("bm25:", p.bm25_corpus_path)
print("parent:", p.parent_corpus_path)
PY
```

Expected:

- `versioned: False`
- `chroma: data/chroma`
- `bm25: data/chroma/bm25_corpus.jsonl`
- `parent: data/chroma/parent_corpus.jsonl`

- [ ] Rebuild local index.

```bash
.venv/bin/python -m scripts.ingest
```

- [ ] Check corpus status.

```bash
.venv/bin/python - <<'PY'
import json
from app.api.routes import build_corpus_status
print(json.dumps(build_corpus_status(), ensure_ascii=False, indent=2))
PY
```

Expected:

- `ready=true`
- `chunk_count > 0`
- `parent_chunk_count > 0`
- `bm25_ready=true`
- Chroma chunk count > 0

- [ ] Optional versioned smoke.

```bash
VERSIONED_INDEXING_ENABLED=true DOCUMENT_INDEX_VERSION=smoke-index .venv/bin/python -m scripts.ingest
VERSIONED_INDEXING_ENABLED=true DOCUMENT_INDEX_VERSION=smoke-index .venv/bin/python - <<'PY'
import json
from app.api.routes import build_corpus_status
print(json.dumps(build_corpus_status(), ensure_ascii=False, indent=2))
PY
```

Expected:

- Versioned `data/indexes/smoke-index` has Chroma, BM25, and parent corpus.
- `ready=true` in versioned mode.

### Acceptance

- Default local app can answer again without setting enterprise/versioned env vars.
- Versioned indexing can still be enabled intentionally.

---

## Phase E: Regenerate Trustworthy Reports

**Problem:** Latest answer evaluation report is not valid quality evidence because the runtime index was empty.

**Decision:** Delete nothing automatically. Generate new reports with fixed runtime paths and mark old zero-score report as obsolete in docs.

### Files likely touched

- `reports/`
- `docs/evaluation.md`
- `docs/interview-guide.md`
- `docs/production-readiness.md`

### Tasks

- [ ] Validate dataset.

```bash
.venv/bin/python -m scripts.evaluate --validate-dataset --check-source-files
```

- [ ] Run retrieval evaluation using runtime-aligned paths.

```bash
.venv/bin/python -m scripts.evaluate --variant hybrid-rerank-parent --top-k 5
```

Expected:

- Positive hit rate is close to previous healthy report.
- Negative rejection remains high.
- Report config includes effective index paths.

- [ ] Run retrieval comparison.

```bash
.venv/bin/python -m scripts.evaluate --compare --top-k 5
```

- [ ] Run answer eval.

If RAGAS credentials are available:

```bash
.venv/bin/python -m scripts.evaluate_answers --limit 10 --sample mixed
```

If judge API is unavailable, run local diagnostics only and label it clearly:

```bash
.venv/bin/python -m scripts.evaluate_answers --limit 10 --sample mixed --no-ragas
```

- [ ] Inspect answer report.
  - `pass_rate` should no longer be `0.0`.
  - `citation_rate` should no longer be `0.0`.
  - `unknown_answer_rate` should not be `1.0` unless the selected cases are all negative.

- [ ] Update docs.
  - Replace references to the zero-score `reports/answer-eval-20260623-134526.json`.
  - State the new report path and exact generated date.
  - If `--no-ragas` was used, explicitly say RAGAS was skipped and not a judge-quality result.

### Acceptance

- Latest retrieval and answer reports reflect the same index used by `/ask`.
- Docs do not cite obsolete or path-mismatched reports as evidence.

---

## Phase F: Final Regression And Smoke

### Commands

Run focused index/eval tests:

```bash
.venv/bin/pytest \
  tests/test_index_versions.py \
  tests/test_api_retriever_factory.py \
  tests/test_evaluation.py \
  tests/test_answer_evaluation.py \
  tests/test_evaluation_endpoint.py \
  -v
```

Run enterprise regression:

```bash
.venv/bin/pytest \
  tests/test_auth_context.py \
  tests/test_acl.py \
  tests/test_permission_ingestion.py \
  tests/test_permission_retrieval.py \
  tests/test_wecom_signature.py \
  tests/test_wecom_routes.py \
  tests/test_wecom_handlers.py \
  tests/test_wecom_client.py \
  tests/test_audit_log.py \
  tests/test_feedback_api.py \
  tests/test_ingestion_jobs.py \
  tests/test_enterprise_smoke.py \
  tests/test_error_handling.py \
  tests/test_metrics.py \
  -v
```

Run full suite:

```bash
.venv/bin/pytest -q
node --test tests/js/ui-utils.test.mjs
```

Manual API smoke:

```bash
make dev
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/corpus/status
curl -sS -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"比亚迪 2024 年年度报告披露的营业额是多少？","top_k":4}' | head -c 1000
```

### Acceptance

- Full Python and JS tests pass.
- `/corpus/status` reports `ready=true`.
- `/ask` returns non-empty answer and non-empty sources for a known positive corpus question.
- Latest reports are regenerated after the fix.
- Final summary clearly separates:
  - implemented and tested code,
  - regenerated quality evidence,
  - remaining production/staging work.

---

## Suggested Commit Sequence

1. `fix: make versioned indexing explicit`
2. `fix: align evaluation retrievers with runtime index paths`
3. `feat: report corpus readiness state`
4. `docs: refresh evaluation evidence after index alignment`

Do not combine regenerated reports with unrelated enterprise feature changes unless explicitly requested.

---

## Known Follow-Ups After This Plan

These are important but lower priority than restoring runtime/evaluation consistency:

- Real enterprise WeChat staging callback verification.
- Per-user rate limiting.
- Move SQLite audit/job storage to a real database.
- Replace local Chroma/BM25 JSONL with production search/vector services.
- Add OpenTelemetry spans for retrieval, rerank, prompt assembly, LLM, and worker jobs.
