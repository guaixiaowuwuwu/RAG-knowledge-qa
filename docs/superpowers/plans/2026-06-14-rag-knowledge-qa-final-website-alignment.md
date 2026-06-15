# RAG Knowledge QA Final Website Alignment Plan

> **For agentic workers:** use the Superpowers workflow when available. Clarify before changing scope, work task by task, prefer tests before implementation, and verify with commands before marking items complete.

**Goal:** Bring this project into close alignment with the `agent-interview-hub` "项目一：RAG 知识问答系统" template while keeping claims evidence-based. The final result should be demoable, measurable, and defensible in an interview: a working RAG system, a credible evaluation loop, realistic technical tradeoffs, deployment documentation, and interview-ready project narration backed by actual artifacts.

**Current baseline:** The repository already has FastAPI, local static frontend, Chroma dense retrieval, BM25 sparse retrieval, RRF fusion, mandatory BGE reranking, parent-child retrieval, Query Rewrite/HyDE, SSE streaming, citations, SEC/BYD public datasets, and lightweight retrieval metrics. The next stage is not a rewrite; it is an alignment and evidence-building stage.

**Non-negotiable rule:** Do not copy the website template's example metrics such as "召回率提升 30%+" or "Top-5 准确率 72% -> 89%" unless this repository produces those numbers from a reproducible experiment. Interview claims must be traceable to code, dataset, report, logs, or documented manual evaluation.

---

## Alignment Targets

### Must Match The Template

- Enterprise knowledge-base RAG positioning.
- Offline ingestion flow: document parsing, cleaning, chunking, embedding, index storage.
- Online QA flow: query preprocessing, hybrid retrieval, RRF fusion, reranking, prompt assembly, LLM generation, streaming answer, citations.
- Multiple document formats: PDF, DOCX, HTML, Markdown, TXT.
- Parent-child indexing.
- Query rewrite and HyDE.
- Reranking.
- Evaluation pipeline with retrieval and answer-level metrics.
- Frontend demo for ingestion, evaluation, streaming QA, and citation inspection.
- Interview-ready documentation explaining technical choices, tradeoffs, failures, and measured results.

### Should Be Clearly Marked As Local-Dev Variant

- Chroma can remain the default local vector store.
- Milvus can be documented and optionally supported through a separate adapter or deployment profile.
- Redis/K8s/Nginx can be documented and optionally scaffolded, but must not be presented as production traffic proof unless actually deployed and tested.

---

## Definition Of Done

The project is "fully aligned" when all of these are true:

- `README.md` explains the full system, setup, demo flow, evaluation flow, and interview talking points.
- A final architecture document exists with offline and online data-flow diagrams.
- The app can ingest a realistic corpus and answer questions with streaming citations.
- There is a labeled evaluation dataset with enough cases to support meaningful retrieval comparison.
- Evaluation can compare at least these variants:
  - dense only
  - dense + BM25 + RRF
  - dense + BM25 + RRF + reranker
  - parent-child on/off
  - query expansion on/off
- The report outputs actual measured metrics and saves them under `reports/`.
- Answer-level evaluation exists as a required RAGAS-based automated pipeline, with deterministic local diagnostics kept as supporting signals.
- Table handling is improved beyond plain paragraph extraction.
- Production-readiness docs exist for Docker, Milvus, Redis cache, rate limits, fallback strategy, and observability.
- Test commands and demo commands are documented and have been run.

---

## Phase 5: Truth Alignment And Reproducible Baseline

**Goal:** Remove ambiguity before adding heavier features. The project should run predictably and not overclaim.

**Files likely touched:**

- `app/core/config.py`
- `.env.example`
- `README.md`
- `pyproject.toml`
- `tests/test_phase5_config.py`
- `docs/architecture.md`

### Tasks

- [ ] Decide the real default embedding behavior.
  - Option A: default to local `bge-m3`, matching current README.
  - Option B: default to `text-embedding-3-small`, then change README to say local BGE is recommended.
  - Acceptance: config, `.env.example`, and README no longer contradict each other.

- [ ] Add a config regression test.
  - Test `Settings().embedding_model`.
  - Test advanced defaults for reranker, parent chunks, query expansion, and eval path.

- [ ] Add `docs/architecture.md`.
  - Include offline ingestion flow.
  - Include online QA flow.
  - Include component table: loader, chunker, embeddings, Chroma, BM25, RRF, reranker, LLM, frontend.
  - Explicitly state local-dev vs production variants.

- [ ] Add a "claims policy" section to README.
  - Explain that all reported metrics are generated from `scripts.evaluate` or answer-level evaluation.
  - Remove or avoid unverified production phrases such as daily QPS, P95 latency, or percentage gains.

### Verification

```bash
.venv/bin/pytest tests/test_phase5_config.py -v
node --test tests/js/ui-utils.test.mjs
```

---

## Phase 6: Evaluation Dataset And Retrieval Benchmark

**Goal:** Build the evidence spine. This is the highest-priority gap for interviews.

**Files likely touched:**

- `data/eval/*.jsonl`
- `app/evaluation/dataset.py`
- `app/evaluation/metrics.py`
- `app/evaluation/report.py`
- `scripts/evaluate.py`
- `tests/test_evaluation.py`
- `tests/test_advanced_evaluation.py`
- `reports/`

### Dataset Tasks

- [ ] Expand the evaluation schema.
  - Required fields: `id`, `question`, `ground_truth`, `expected_sources`, `expected_answer_keywords`.
  - Optional fields: `category`, `difficulty`, `language`, `notes`.

- [ ] Build at least 50 initial labeled QA cases.
  - Include Chinese BYD policy/report questions.
  - Include English SEC filing questions.
  - Include exact-match questions, summary questions, comparison questions, and "not in corpus" questions.

- [ ] Add dataset validation.
  - Detect missing source files.
  - Detect empty ground truth.
  - Detect duplicate IDs.
  - Detect questions with no expected source unless explicitly marked as negative.

### Benchmark Tasks

- [ ] Add retriever variant selection.
  - CLI examples:
    - `--variant dense`
    - `--variant hybrid`
    - `--variant hybrid-rerank`
    - `--variant hybrid-rerank-parent`
    - `--variant full`

- [ ] Add retrieval metrics.
  - Hit Rate@K
  - MRR@K
  - Source Recall@K
  - optionally Precision@K and NDCG@K

- [ ] Save reports to `reports/retrieval-YYYYMMDD-HHMMSS.json`.
  - Include config snapshot.
  - Include dataset path and case count.
  - Include per-case retrieved sources.
  - Include summary metrics.

- [ ] Add comparison report generation.
  - One command should run all variants and produce a table.
  - README should show the latest real table, not template numbers.

### Verification

```bash
.venv/bin/python -m scripts.evaluate --variant full --top-k 5
.venv/bin/python -m scripts.evaluate --compare --top-k 5
.venv/bin/pytest tests/test_evaluation.py tests/test_advanced_evaluation.py -v
```

---

## Phase 7: Answer-Level Evaluation

**Goal:** Move beyond retrieval-only metrics and evaluate whether answers are grounded, relevant, and complete.

**Files likely touched:**

- `app/evaluation/answer_eval.py`
- `scripts/evaluate_answers.py`
- `pyproject.toml`
- `.env.example`
- `tests/test_answer_evaluation.py`
- `reports/`

### Tasks

- [ ] Implement answer evaluation records.
  - Store question, ground truth, generated answer, retrieved contexts, sources, and model config.

- [ ] Add a local deterministic diagnostic evaluator.
  - Keyword coverage from `expected_answer_keywords`.
  - Citation presence.
  - Empty/unknown answer handling.
  - Negative-case behavior.

- [ ] Add required RAGAS integration as the answer-quality pipeline.
  - Metrics target:
    - faithfulness
    - answer relevancy
    - context precision
    - context recall
  - Build a RAGAS `Dataset` with `question`, `answer`, `contexts`, and `ground_truth`.
  - Use explicit RAGAS judge LLM and embedding configuration.
  - Fail fast when RAGAS dependencies, judge credentials, judge-model calls, or embedding calls are unavailable.
  - Save partial answer records on failure for diagnosis.

- [ ] Add `scripts.evaluate_answers`.
  - Support `--limit`.
  - Support `--output reports/answer-eval-*.json`.
  - Run RAGAS by default and write metrics to the report.

- [ ] Add a README "How to talk about evaluation" section.
  - Explain retrieval metrics vs answer metrics.
  - Explain what improved and what did not.
  - Include known limitations.

### Verification

```bash
.venv/bin/python -m scripts.evaluate_answers --limit 10
.venv/bin/pytest tests/test_answer_evaluation.py -v
```

---

## Phase 8: Document Understanding Upgrade

**Goal:** Better align with the template's "PDF/Word/HTML/table/image" promise without pretending full multimodal production support exists.

**Files likely touched:**

- `app/ingestion/loaders.py`
- `app/ingestion/table_extractors.py`
- `app/ingestion/cleaning.py`
- `app/ingestion/chunker.py`
- `tests/test_table_extraction.py`
- `tests/test_document_cleaning.py`
- `pyproject.toml`

### Tasks

- [ ] Add document cleaning.
  - Normalize whitespace.
  - Remove repeated headers/footers where detectable.
  - Preserve page metadata.
  - Preserve section headings when possible.

- [ ] Improve DOCX parsing.
  - Include tables, not just paragraphs.
  - Convert tables to Markdown.
  - Preserve heading hierarchy as metadata where feasible.

- [ ] Improve HTML parsing.
  - Prefer semantic content extraction.
  - Preserve headings and table text.
  - Avoid script/style/navigation noise.

- [ ] Add PDF table handling.
  - Start with a lightweight extractor or optional dependency.
  - Convert detected tables to Markdown blocks.
  - Mark table chunks with `content_type=table`.

- [ ] Add scanned PDF/OCR as optional.
  - Document this as optional and slower.
  - Do not block normal ingestion if OCR tools are absent.

- [ ] Add table-aware chunking.
  - Keep small tables whole.
  - Avoid splitting table rows across chunks when possible.

### Verification

```bash
.venv/bin/pytest tests/test_loaders.py tests/test_docx_html_loaders.py tests/test_table_extraction.py -v
.venv/bin/python -m scripts.ingest
```

---

## Phase 9: Retrieval Quality Controls

**Goal:** Make the advanced RAG behavior configurable, observable, and explainable.

**Files likely touched:**

- `app/rag/hybrid_retriever.py`
- `app/rag/query_transform.py`
- `app/rag/service.py`
- `app/api/schemas.py`
- `app/web/static/*`
- `tests/test_hybrid_retriever.py`
- `tests/test_query_transform.py`

### Tasks

- [ ] Return retrieval debug metadata when requested.
  - Query variants used.
  - Dense candidates.
  - BM25 candidates.
  - RRF scores.
  - Reranker scores.
  - Parent hydration info.

- [ ] Add API-level `debug` option.
  - Default false.
  - When true, include retrieval trace in response.

- [ ] Add frontend debug panel.
  - Show query variants.
  - Show top retrieved chunks and scores.
  - Keep it collapsed by default.

- [ ] Add configurable toggles.
  - Enable/disable rewrite.
  - Enable/disable HyDE.
  - Enable/disable parent hydration.
  - Enable/disable reranker for benchmark variants only.

- [ ] Add guardrails for query expansion.
  - Limit variants.
  - Timeout LLM calls where possible.
  - Fall back to original query if transformation fails.

### Verification

```bash
.venv/bin/pytest tests/test_hybrid_retriever.py tests/test_query_transform.py tests/test_rag_service.py -v
```

---

## Phase 10: Frontend Demo Polish

**Goal:** Make the UI credible for interviews and demos, not just functional.

**Files likely touched:**

- `app/web/static/index.html`
- `app/web/static/styles.css`
- `app/web/static/app.js`
- `app/web/static/ui-utils.js`
- `tests/js/ui-utils.test.mjs`
- `tests/test_web_static.py`

### Tasks

- [ ] Add evaluation report view.
  - Show latest metrics.
  - Show per-case hits/misses.
  - Show variant comparison if available.

- [ ] Add corpus status view.
  - Document count.
  - Chunk count.
  - Parent chunk count.
  - BM25 corpus status.
  - Chroma collection name.

- [ ] Improve citation cards.
  - Highlight source file, page, chunk, content type.
  - Support long content without layout shift.

- [ ] Add example question presets.
  - BYD Chinese examples.
  - SEC English examples.
  - Negative example.

- [ ] Add visual states.
  - Loading, empty, error, complete.
  - Streaming answer state should stay stable.

### Verification

```bash
node --test tests/js/ui-utils.test.mjs
.venv/bin/pytest tests/test_web_static.py tests/test_evaluation_endpoint.py -v
```

Manual browser check:

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` and verify ingestion, evaluation, streaming answer, citations, and responsive layout.

---

## Phase 11: Production-Readiness Scaffold

**Goal:** Align with the website template's production deployment discussion without claiming untested scale.

**Files likely touched:**

- `Dockerfile`
- `docker-compose.yml`
- `deploy/`
- `docs/deployment.md`
- `docs/production-readiness.md`
- `app/core/config.py`
- `app/rag/cache.py`
- `app/middleware.py`
- `tests/test_cache.py`

### Tasks

- [x] Add Dockerfile for the FastAPI service.

- [x] Add Docker Compose local stack.
  - API service.
  - Optional Redis.
  - Optional Milvus or Qdrant profile.

- [x] Add vector-store abstraction notes.
  - Current: Chroma.
  - Production option: Milvus.
  - Explain migration path and index tradeoffs.

- [x] Add Redis cache abstraction.
  - Cache final answers or retrieval results.
  - Include TTL.
  - Include cache key design.
  - Keep disabled by default.

- [x] Add request limits and timeouts.
  - `top_k` already bounded.
  - Add question length limit if missing.
  - Document LLM timeout behavior.

- [x] Add fallback strategy documentation.
  - LLM API unavailable.
  - Reranker unavailable.
  - Empty retrieval.
  - Index not built.

- [x] Add observability documentation.
  - Structured logs.
  - Latency timings for retrieval, rerank, LLM.
  - Token usage if available.
  - Suggested LangSmith/Phoenix integration.

- [x] Add a lightweight latency benchmark script.
  - Do not claim production QPS from local laptop results.
  - Save local benchmark output under `reports/`.

### Verification

```bash
docker compose up --build
curl http://127.0.0.1:8000/health
.venv/bin/python -m scripts.benchmark --limit 20
```

---

## Phase 12: Final Interview Package

**Goal:** Turn the engineering work into a defensible interview artifact.

**Files likely touched:**

- `README.md`
- `docs/interview-guide.md`
- `docs/architecture.md`
- `docs/evaluation.md`
- `docs/tradeoffs.md`
- `reports/latest-*.json`

### Tasks

- [ ] Write `docs/interview-guide.md`.
  - 1-minute version.
  - 3-minute version.
  - 5-minute version.
  - Architecture walkthrough.
  - What I built personally.

- [ ] Write `docs/tradeoffs.md`.
  - Why Chroma locally.
  - Why BM25 + dense.
  - Why RRF.
  - Why reranker.
  - Why parent-child retrieval.
  - Why Query Rewrite/HyDE.
  - When not to use each technique.

- [ ] Write `docs/evaluation.md`.
  - Dataset construction.
  - Metrics definitions.
  - Variant comparison table.
  - Failure cases.
  - Next optimization ideas.

- [ ] Add "common interview questions" answers.
  - 分块大小怎么选？
  - 检索效果不好怎么排查？
  - 为什么需要 BM25？
  - Reranker 带来什么代价？
  - 如何评估 RAG？
  - 如何处理表格和图片？
  - 如何部署到生产？
  - 如何控制成本和延迟？

- [ ] Add a final demo script.
  - Setup.
  - Ingest BYD corpus.
  - Run evaluation.
  - Ask three successful questions.
  - Ask one negative question.
  - Show citations and debug trace.

### Verification

```bash
.venv/bin/pytest -q
node --test tests/js/ui-utils.test.mjs
.venv/bin/python -m scripts.ingest
.venv/bin/python -m scripts.evaluate --compare --top-k 5
.venv/bin/python -m scripts.evaluate_answers --limit 10
```

---

## Recommended Execution Order

1. Phase 5: fix truth/config/docs.
2. Phase 6: build evaluation dataset and retrieval benchmark.
3. Phase 7: add answer-level evaluation.
4. Phase 8: improve document/table parsing.
5. Phase 9: add retrieval observability and toggles.
6. Phase 10: polish frontend demo.
7. Phase 11: add production-readiness scaffold.
8. Phase 12: write final interview package.

Reason: evaluation must come before optimization claims. Otherwise the project can look feature-rich but remain hard to defend under interview pressure.

---

## Milestone Acceptance Checklist

- [ ] `pytest` passes in the project virtualenv.
- [ ] JS tests pass.
- [ ] Ingestion works on BYD corpus.
- [ ] Ingestion works on SEC corpus or a documented subset.
- [ ] `/ask` returns citations.
- [ ] `/ask/stream` streams tokens and then sources.
- [ ] `/evaluation/report` returns real metrics.
- [ ] A comparison report exists under `reports/`.
- [ ] Answer-level evaluation report exists under `reports/`.
- [ ] README does not contain unsupported metrics.
- [ ] Docs explain local vs production architecture.
- [ ] Interview guide includes real measured numbers from the latest report.

---

## Risks And Mitigations

- **Risk:** RAGAS setup becomes slow or fragile.
  - **Mitigation:** make RAGAS failures explicit, keep deterministic local diagnostics in the report, save partial failure reports, and run smaller `--limit` smoke checks before full scheduled evaluation.

- **Risk:** Large local models make demo slow.
  - **Mitigation:** provide warmup script, smaller demo corpus, and clear `.env` profiles.

- **Risk:** Evaluation dataset is too small to support claims.
  - **Mitigation:** minimum 50 cases for first report; target 100-200 before final interview use.

- **Risk:** Table/OCR support expands scope too much.
  - **Mitigation:** make table extraction required, OCR optional.

- **Risk:** Production scaffold distracts from core RAG quality.
  - **Mitigation:** do production docs after evaluation and document understanding, not before.

---

## Final Deliverables

- `docs/architecture.md`
- `docs/evaluation.md`
- `docs/tradeoffs.md`
- `docs/interview-guide.md`
- `reports/retrieval-comparison-*.json`
- `reports/answer-eval-*.json`
- Updated `README.md`
- Optional `Dockerfile` and `docker-compose.yml`
- Tests covering config, evaluation, loaders, retrieval, answer evaluation, API, and frontend utilities
