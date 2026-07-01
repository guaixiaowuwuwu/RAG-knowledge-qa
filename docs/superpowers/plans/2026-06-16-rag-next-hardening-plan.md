# RAG Knowledge QA Next Hardening Plan

> **Context:** As of 2026-06-16, the project has a clean git worktree, 95 passing Python tests, 13 passing JS tests, a validated 60-case eval dataset, retrieval comparison reports, answer-eval/RAGAS smoke reports, table extraction, debug traces, Docker/Compose, cache scaffold, and interview docs. This plan focuses only on the next high-leverage gaps found during the latest review.

**Goal:** Turn the current project from "feature-complete and demoable" into "harder to challenge in interviews" by strengthening refusal behavior, answer-level evidence, evaluation granularity, and report analysis.

**Do not do yet:** broad refactors, new frameworks, full vector DB migration, auth/multi-tenant work, or production SLA claims. Those are lower priority than fixing the evidence and refusal gaps.

---

## Priority Order

1. Fix negative/unknown-question behavior.
2. Expand answer-level RAGAS evaluation beyond smoke tests.
3. Add page/chunk-level evaluation evidence.
4. Add grouped evaluation reports.
5. Clean up small corpus-status serialization issue.

This order matters: refusal quality and answer evidence are the biggest interview risks right now.

---

## Phase A: Negative Refusal And Confidence Gating

**Problem:** Current retrieval reports show `negative_rejection_rate=0.000` across variants. Negative questions such as today's stock price, real-time factory output, private contracts, or non-corpus companies still retrieve surface-related chunks.

**Target outcome:** The system should not treat every non-empty retrieval result as sufficient evidence. It should detect low-confidence or out-of-scope questions and return a grounded refusal with no misleading citation when appropriate.

### Files likely touched

- `app/rag/hybrid_retriever.py`
- `app/rag/service.py`
- `app/rag/prompts.py`
- `app/core/config.py`
- `.env.example`
- `app/api/schemas.py`
- `app/evaluation/metrics.py`
- `app/evaluation/report.py`
- `tests/test_hybrid_retriever.py`
- `tests/test_rag_service.py`
- `tests/test_evaluation.py`

### Tasks

- [x] Add confidence settings.
  - `MIN_RERANKER_SCORE`
  - `MIN_FINAL_SOURCE_COUNT`
  - `ENABLE_LOW_CONFIDENCE_REFUSAL`
  - `TIME_SENSITIVE_REFUSAL_ENABLED`

- [x] Add a lightweight intent/rule detector for unsupported real-time or private-data questions.
  - Chinese examples: `今天`, `此刻`, `实时`, `当前股价`, `收盘价`, `私密合同`, `未披露`
  - English examples: `today`, `current stock price`, `market close today`, `real-time`, `private contract`, `audited net income` for non-corpus entities

- [x] Add a retrieval-confidence decision object.
  - Store final source count.
  - Store best reranker score when available.
  - Store whether final chunks contain expected entity/source hints when possible.
  - Store refusal reason: `time_sensitive`, `private_or_unavailable`, `low_retrieval_confidence`, `empty_retrieval`.

- [x] Update `RagService.answer`.
  - If confidence decision refuses, return a clear "知识库中没有找到可支持该问题的资料" style answer.
  - For refusal, return no sources unless the source is explicitly used to explain absence.
  - Include refusal reason in debug trace.

- [x] Update streaming behavior.
  - SSE should emit refusal token(s), empty sources, and debug payload.

- [x] Update prompt to reinforce refusing unsupported claims.
  - Keep prompt concise.
  - Make it clear that real-time/current-market/private information cannot be inferred from annual reports.

- [x] Add tests for all negative examples in `data/eval/sample_eval.jsonl`.
  - Expected: no citations or explicit refusal marker.
  - Ensure positive cases still return sources.

### Verification

```bash
.venv/bin/pytest tests/test_rag_service.py tests/test_hybrid_retriever.py tests/test_evaluation.py -v
.venv/bin/python -m scripts.evaluate --variant hybrid-rerank-parent --top-k 5
```

### Acceptance

- `negative_rejection_rate` should improve from `0.000`.
- Positive-case `hit_rate@5` should not collapse.
- Refusal cases should be visible in debug output with reason codes.

**Completed 2026-06-16:** `reports/retrieval-20260616-125153.json` shows `negative_rejection_rate=1.000`, `hit_rate@5=0.981`, and refusal reasons `private_or_unavailable=5`, `time_sensitive=3`.

---

## Phase B: Expand RAGAS Answer-Level Evidence

**Problem:** Current RAGAS reports are smoke checks with only 1 case. They prove the pipeline runs, but not answer quality.

**Target outcome:** Produce a representative answer-eval report that can be cited carefully in interviews.

### Files likely touched

- `scripts/evaluate_answers.py`
- `app/evaluation/answer_eval.py`
- `docs/evaluation.md`
- `docs/interview-guide.md`
- `reports/`
- `tests/test_answer_evaluation.py`

### Tasks

- [x] Add answer-eval case sampling.
  - `--sample mixed`
  - `--include-categories exact_fact,summary,policy,sec_filing,comparison,risk,negative`
  - `--language zh|en|all`
  - Keep deterministic ordering unless `--random-seed` is provided.

- [x] Add `--no-ragas` mode only for local diagnosis.
  - Default should still run RAGAS if credentials are present.
  - Failure should save partial report and exit non-zero.

- [x] Run at least a 10-case representative RAGAS report.
  - Include both Chinese and English cases.
  - Include at least one negative case after Phase A refusal gating.

- [x] Add report summary extraction.
  - Print local summary.
  - Print RAGAS summary.
  - Save generated report path.

- [x] Update `docs/evaluation.md`.
  - Replace "only 1 sample smoke" language after a real 10+ case report exists.
  - Keep the caveat if the report is still small.

- [x] Update `docs/interview-guide.md`.
  - Include only metrics from the latest real report.
  - Add a short note on judge model and reproducibility.

### Verification

```bash
.venv/bin/python -m scripts.evaluate_answers --limit 10
.venv/bin/pytest tests/test_answer_evaluation.py -v
```

### Acceptance

- A new `reports/answer-eval-*.json` exists with at least 10 cases.
- It includes `summary`, `records`, and `ragas`.
- Docs no longer rely on a 1-case smoke report as the main answer-quality artifact.

**Completed 2026-06-16:** `reports/answer-eval-20260616-132709.json` covers 10 mixed cases with RAGAS enabled. Local `pass_rate=0.800`, `negative_case_pass_rate=1.000`; RAGAS `faithfulness=0.867`, `answer_relevancy=0.783`, `context_precision=0.653`, `context_recall=0.700`. Judge model: `deepseek-v4-pro`; embedding: local `bge-m3`.

---

## Phase C: Page/Chunk-Level Retrieval Evaluation

**Problem:** Current evaluation is source-file level. It can count as a hit even when retrieval only finds the right document but not the right evidence.

**Target outcome:** Add stricter evidence-level evaluation without discarding the existing source-level metrics.

### Files likely touched

- `data/eval/sample_eval.jsonl`
- `app/evaluation/dataset.py`
- `app/evaluation/metrics.py`
- `app/evaluation/report.py`
- `scripts/evaluate.py`
- `tests/test_evaluation.py`
- `tests/test_advanced_evaluation.py`
- `docs/evaluation.md`

### Tasks

- [x] Extend eval schema with optional fields.
  - `expected_pages`
  - `expected_chunk_keywords`
  - `evidence_notes`

- [x] Add parser support while keeping backward compatibility.

- [x] Add metrics.
  - `page_hit_rate_at_k`
  - `evidence_keyword_recall_at_k`
  - `evidence_strict_hit_at_k`

- [x] Annotate an initial subset.
  - Start with 15-20 cases.
  - Prioritize exact financial facts, policy clauses, and SEC risk disclosures.

- [x] Add report fields.
  - Per-case page hit.
  - Per-case evidence keyword matches/misses.
  - Summary strict metrics.

- [x] Update docs with interpretation.
  - Source-level metrics are broad recall.
  - Page/chunk/evidence metrics are stronger proof.

### Verification

```bash
.venv/bin/python -m scripts.evaluate --validate-dataset --check-source-files
.venv/bin/python -m scripts.evaluate --variant dense --top-k 5
.venv/bin/pytest tests/test_evaluation.py tests/test_advanced_evaluation.py -v
```

### Acceptance

- Existing 60-case dataset still loads.
- At least 15 cases have stricter evidence annotations.
- Reports show both broad source-level and stricter evidence-level metrics.

**Completed 2026-06-16:** `data/eval/sample_eval.jsonl` has 18 evidence-annotated cases. `reports/retrieval-20260616-132031.json` shows broad source metrics and strict evidence metrics: `hit_rate@5=0.962`, `page_hit_rate@5=0.625`, `evidence_keyword_recall@5=0.507`, `evidence_strict_hit@5=0.278`, `negative_rejection_rate=1.000`.

---

## Phase D: Grouped Evaluation Reports

**Problem:** Overall averages hide useful behavior. Dense is strongest overall right now, but the project needs category-level insight to explain tradeoffs.

**Target outcome:** Reports should show performance by language, category, difficulty, and positive/negative status.

### Files likely touched

- `app/evaluation/report.py`
- `scripts/evaluate.py`
- `app/api/routes.py`
- `app/api/schemas.py`
- `app/web/static/app.js`
- `app/web/static/index.html`
- `app/web/static/styles.css`
- `tests/test_evaluation.py`
- `tests/test_evaluation_endpoint.py`
- `tests/js/ui-utils.test.mjs`

### Tasks

- [x] Add grouped summary generation.
  - Group by `language`.
  - Group by `category`.
  - Group by `difficulty`.
  - Group by `is_negative`.

- [x] Add grouped summaries to JSON report.
  - Preserve current `summary` structure.
  - Add `groups`.

- [x] Update comparison report.
  - Include group-level metrics per variant.
  - Keep markdown table concise.

- [x] Update API response schemas.

- [x] Update frontend evaluation panel.
  - Show overall metrics first.
  - Add grouped metric table below.
  - Keep UI compact and readable.

### Verification

```bash
.venv/bin/python -m scripts.evaluate --compare --top-k 5
.venv/bin/pytest tests/test_evaluation.py tests/test_evaluation_endpoint.py -v
node --test tests/js/ui-utils.test.mjs
```

### Acceptance

- Reports explain where dense/hybrid/rerank/parent help or hurt.
- Interview guide can cite category-specific observations instead of one overall number.

**Completed 2026-06-16:** `reports/retrieval-comparison-20260616-141357.json` includes `groups` for each local comparison variant. Overall: dense `hit_rate@5=0.962`, hybrid/hybrid-rerank/hybrid-rerank-parent `hit_rate@5=1.000`, all local variants `negative_rejection_rate=1.000`. Group observations are documented in `docs/evaluation.md` and `docs/interview-guide.md`.

---

## Phase E: Corpus Status Serialization Cleanup

**Problem:** `/corpus/status` works through FastAPI, but direct `json.dumps(build_corpus_status())` fails because the returned dict contains Pydantic response model objects.

**Target outcome:** Internal function returns plain JSON-serializable data, while endpoint behavior remains unchanged.

### Files likely touched

- `app/api/routes.py`
- `tests/test_evaluation_endpoint.py` or new `tests/test_corpus_status.py`

### Tasks

- [x] Add a regression test.
  - Call `build_corpus_status()`.
  - Assert `json.dumps(..., ensure_ascii=False)` succeeds.
  - Assert endpoint still returns expected fields.

- [x] Convert nested Pydantic objects to plain dicts.
  - Use `.model_dump()` or return plain dicts from `jsonl_status` / `chroma_collection_status`.

- [x] Keep response model validation intact.

### Completed Notes

- `build_corpus_status()` now returns JSON-serializable nested dicts for BM25, parent corpus, and Chroma status.
- Added regression coverage that directly serializes the internal payload and still exercises `/corpus/status` through FastAPI response-model validation.

### Verification

```bash
.venv/bin/pytest tests/test_evaluation_endpoint.py -v
.venv/bin/python - <<'PY'
import json
from app.api.routes import build_corpus_status
print(json.dumps(build_corpus_status(), ensure_ascii=False)[:200])
PY
```

### Acceptance

- Direct serialization works.
- `/corpus/status` still returns 200.

---

## Final Verification For This Plan

Run this after Phases A-E:

```bash
.venv/bin/pytest -q
node --test tests/js/ui-utils.test.mjs
.venv/bin/python -m scripts.evaluate --validate-dataset --check-source-files
.venv/bin/python -m scripts.evaluate --compare --top-k 5
.venv/bin/python -m scripts.evaluate_answers --limit 10
```

Expected final artifacts:

- Updated retrieval comparison report under `reports/`.
- New answer-eval report with 10+ cases and RAGAS metrics.
- Evaluation report includes grouped metrics.
- Evaluation dataset includes evidence-level annotations for a subset.
- Docs updated with latest real numbers and caveats.

---

## Interview-Safe Claims After Completion

Only make claims supported by the new reports.

Safe claim pattern:

> "I implemented dense, BM25, RRF, reranker, parent-child retrieval, and query expansion as comparable variants. On my 60-case public-company eval set, the current best variant is X on metric Y. The report also exposed weakness Z, so I added refusal gating and track negative rejection separately."

Unsafe claim pattern:

> "Hybrid retrieval improved recall by 30% and the system is production-ready."

Do not use the unsafe version unless the repository contains a report proving it.
