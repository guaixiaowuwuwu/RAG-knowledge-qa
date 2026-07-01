# RAG 企业微信生产落地执行计划

> **For Codex / agentic workers:** 这是一份可执行 plan。按阶段顺序推进；每个任务先写测试或最小验证，再改实现。不要一次性重构全部 RAG 链路。企业微信接口细节在实现前必须核对企业微信官方最新文档，避免把过期字段写死。

**Goal:** 把当前本地 RAG 知识问答系统升级为一个可在企业内试点落地的系统：支持企业微信入口、用户身份、文档权限过滤、问答审计、异步入库、索引版本和基础生产观测。

**Current baseline:** 仓库已有 FastAPI、`/ask`、`/ask/stream`、同步 `/ingest`、Chroma、本地 BM25 JSONL、parent corpus JSONL、BGE reranker、Query Rewrite/HyDE、Redis 答案缓存抽象、静态前端、评估脚本和 Docker Compose。当前缺口是身份认证、权限过滤、企业微信集成、异步任务、索引版本、审计和生产可观测性。

**Critical constraint:** 不允许只做企业微信回调然后直接复用裸 `/ask`。企业项目必须先保证用户身份和文档权限不会串。

---

## Priority Order

1. 增加用户身份、API 鉴权和文档 ACL 数据模型。
2. 在入库、向量检索、BM25、parent hydration、缓存中贯穿权限过滤。
3. 接入企业微信应用入口：OAuth/回调验签/消息收发。
4. 增加问答审计、反馈和后台查询接口。
5. 把同步入库改为异步任务，并加入索引版本和回滚。
6. 增强生产部署、观测、评估和文档。

这个顺序不能反过来。企业微信入口没有权限过滤会放大数据泄露风险。

---

## Phase A: Identity, Auth, Tenant And ACL Foundation

**Problem:** 当前 API 没有用户、租户、部门或权限概念。`/ask`、`/ask/stream`、`/ingest`、评估接口都是裸露入口。

**Target outcome:** 所有业务接口都能拿到一个统一的 `RequestContext`，包含 `tenant_id`、`user_id`、`department_ids`、`roles`、`permission_version`。管理员接口必须鉴权。

### Files likely touched

- `app/core/config.py`
- `.env.example`
- `app/api/routes.py`
- `app/api/schemas.py`
- `app/middleware.py`
- Create: `app/security/context.py`
- Create: `app/security/auth.py`
- Create: `app/security/acl.py`
- Create: `app/storage/sqlite.py`
- Create: `tests/test_auth_context.py`
- Create: `tests/test_acl.py`

### Tasks

- [x] Add settings for enterprise mode.
  - `AUTH_ENABLED=true|false`, default `false` for local dev.
  - `ADMIN_API_KEYS`, comma-separated for local/admin bootstrap.
  - `DEFAULT_TENANT_ID=default`.
  - `AUDIT_DB_PATH=data/runtime/audit.sqlite3`.
  - `PERMISSION_VERSION=local-v1`.

- [x] Create `RequestContext`.
  - Fields: `tenant_id`, `user_id`, `display_name`, `department_ids`, `roles`, `permission_version`, `source`.
  - `source` values: `local`, `api_key`, `wecom`.
  - Local dev fallback only works when `AUTH_ENABLED=false`.

- [x] Add FastAPI dependencies.
  - `get_request_context(request) -> RequestContext`.
  - `require_admin(context)`.
  - `require_authenticated(context)`.

- [x] Protect endpoints.
  - `/ask` and `/ask/stream`: require authenticated context when `AUTH_ENABLED=true`.
  - `/ingest`, `/evaluation/*`, `/corpus/status`: require admin when `AUTH_ENABLED=true`.
  - Keep current no-auth behavior only for local dev mode.

- [x] Create ACL primitives.
  - `DocumentACL`: `tenant_id`, `allowed_user_ids`, `allowed_department_ids`, `allowed_roles`, `is_public`.
  - `can_access_document(context, acl)`.
  - Rule: same tenant required; public allows tenant users; user match, department overlap, or role overlap allows access.

- [x] Add tests.
  - No-auth local mode still works.
  - Auth-enabled request without credentials returns 401.
  - Non-admin cannot call `/ingest`.
  - Admin API key can call admin endpoints.
  - ACL denies cross-tenant access.
  - ACL allows explicit user, department, role, and public tenant access.

### Verification

```bash
.venv/bin/pytest tests/test_auth_context.py tests/test_acl.py tests/test_evaluation_endpoint.py -v
```

### Acceptance

- Business endpoints receive a `RequestContext`.
- Admin endpoints are protected when auth is enabled.
- Local demo flow remains available with `AUTH_ENABLED=false`.
- ACL behavior is deterministic and unit tested.

---

## Phase B: Permission-Aware Ingestion And Retrieval

**Problem:** 当前 chunk metadata 没有企业权限字段，检索结果无法按用户过滤。缓存 key 也没有用户/权限维度。

**Target outcome:** 文档权限随入库写入 child chunks、BM25 corpus 和 parent corpus；检索时按 `RequestContext` 过滤；答案缓存不会跨用户或权限版本复用。

### Files likely touched

- `app/ingestion/loaders.py`
- `app/ingestion/pipeline.py`
- `app/rag/vector_store.py`
- `app/rag/bm25.py`
- `app/rag/parent_store.py`
- `app/rag/hybrid_retriever.py`
- `app/rag/service.py`
- `app/rag/cache.py`
- `app/api/routes.py`
- Create: `app/ingestion/manifest.py`
- Create: `tests/test_permission_ingestion.py`
- Create: `tests/test_permission_retrieval.py`
- Modify: `tests/test_cache.py`

### Tasks

- [x] Add document manifest support.
  - Default path: `data/documents_manifest.json`.
  - Manifest maps document path patterns to ACL metadata.
  - If no manifest entry exists in local dev, use `tenant_id=default`, `is_public=true`.

- [x] Extend chunk metadata during ingestion.
  - Add `tenant_id`, `doc_id`, `document_version`, `allowed_user_ids`, `allowed_department_ids`, `allowed_roles`, `is_public`.
  - Ensure metadata is JSON-serializable and Chroma-compatible.
  - Persist same fields to BM25 JSONL and parent corpus JSONL.

- [x] Add permission filter object.
  - Create `RetrievalAccessFilter` from `RequestContext`.
  - Include `tenant_id`, `user_id`, `department_ids`, `roles`, `permission_version`.

- [x] Filter dense retrieval.
  - Prefer Chroma `where` filters for `tenant_id` and simple public/user fields where supported.
  - Apply a Python post-filter using `can_access_document` for full ACL logic.
  - Over-fetch before filtering: request at least `dense_top_k * 3`, capped by config.

- [x] Filter BM25 retrieval.
  - Store ACL metadata in `RetrievedDocument.metadata`.
  - Apply `can_access_document` before final top-k.

- [x] Filter parent hydration.
  - Parent chunks must be checked against the same ACL.
  - If parent is denied, keep no parent content; do not leak parent text through citations or debug.

- [x] Update `HybridRetriever`.
  - Accept `access_filter` or `request_context`.
  - Pass filter to dense, sparse, and parent hydration.
  - Include permission filter summary in debug without exposing raw secrets.

- [x] Update `RagService`.
  - `answer()` and `answer_stream()` accept `context`.
  - The prompt only receives allowed chunks.

- [x] Update cache key.
  - Include `tenant_id`, `user_id` or stable user hash, `permission_version`, and `document_index_version`.
  - Never cache answers when auth is enabled and no context exists.

- [x] Add tests.
  - Same question returns different sources for users in different departments.
  - Cross-tenant documents never appear in sources, debug final chunks, or SSE sources.
  - BM25-only match still respects ACL.
  - Parent hydration cannot leak denied parent content.
  - Cache keys differ for different users and permission versions.

### Verification

```bash
.venv/bin/pytest tests/test_permission_ingestion.py tests/test_permission_retrieval.py tests/test_cache.py tests/test_rag_service.py -v
```

### Acceptance

- A user can only retrieve documents allowed by ACL.
- Permission filtering applies consistently to dense, BM25, reranked final chunks, parent context, sources, debug payloads, and cache.
- Existing local public-corpus demo still works.

---

## Phase C: WeCom Enterprise WeChat Integration

**Problem:** 当前系统只能通过 Web/API 使用，没有企业微信应用入口，也没有企业微信用户身份映射。

**Target outcome:** 企业微信用户可以从企业微信应用或机器人提问；系统能验证企业微信回调、解析用户、调用 RAG、返回答案和引用摘要。

### Files likely touched

- `app/core/config.py`
- `.env.example`
- `app/main.py`
- Create: `app/integrations/wecom/config.py`
- Create: `app/integrations/wecom/crypto.py`
- Create: `app/integrations/wecom/client.py`
- Create: `app/integrations/wecom/routes.py`
- Create: `app/integrations/wecom/schemas.py`
- Create: `app/integrations/wecom/handlers.py`
- Create: `tests/test_wecom_signature.py`
- Create: `tests/test_wecom_routes.py`
- Create: `tests/test_wecom_handlers.py`

### Tasks

- [x] Verify official docs before coding.
  - Confirm URL verification algorithm, message callback encryption, access token API, user identity API, and send application message API.
  - Record doc links and checked date in `docs/deployment.md` or a new integration doc.

- [x] Add WeCom settings.
  - `WECOM_ENABLED=false`.
  - `WECOM_CORP_ID`.
  - `WECOM_AGENT_ID`.
  - `WECOM_SECRET`.
  - `WECOM_TOKEN`.
  - `WECOM_ENCODING_AES_KEY`.
  - `WECOM_CALLBACK_PATH=/integrations/wecom/callback`.
  - `WECOM_RESPONSE_MODE=passive|active`, default `active`.

- [x] Implement WeCom crypto and callback verification.
  - Verify callback signature.
  - Decrypt incoming XML if encrypted mode is configured.
  - Encrypt passive replies when needed.
  - Unit tests should use deterministic sample payloads stored under `tests/fixtures/wecom/`.

- [x] Implement WeCom API client.
  - Fetch and cache `access_token`.
  - Get user info by OAuth code or message `FromUserName`.
  - Send text or textcard application messages.
  - Add timeout and retry settings.
  - Tests mock HTTP calls with `httpx` or FastAPI test client transport.

- [x] Add user mapping.
  - Store mapping: `tenant_id`, `wecom_userid`, `system_user_id`, `display_name`, `department_ids`, `roles`, `permission_version`.
  - For v1, use SQLite table or JSON mapping file; keep an adapter interface so it can later move to enterprise IAM.
  - Unknown user default: authenticated but no document permissions except tenant public docs.

- [x] Add message handler.
  - Text question -> build `RequestContext(source="wecom")`.
  - Call `RagService.answer` with permission context.
  - Return concise answer plus top citations.
  - If answer is too long, send summary in WeCom and include a URL to Web detail page or audit record.
  - Refusal answer should be sent as-is with no misleading source.

- [x] Add routes.
  - `GET /integrations/wecom/callback`: URL verification.
  - `POST /integrations/wecom/callback`: receive message callback.
  - `GET /integrations/wecom/oauth/callback`: optional web OAuth callback for browser entry.

- [x] Add tests.
  - Invalid signature rejected.
  - Valid callback creates context with correct `wecom_userid`.
  - Text message calls RAG with permission context.
  - Long answer is split or converted to card/link.
  - Upstream WeCom API failure returns a safe response and logs error.

### Verification

```bash
.venv/bin/pytest tests/test_wecom_signature.py tests/test_wecom_routes.py tests/test_wecom_handlers.py -v
```

### Acceptance

- WeCom callback can be verified in local/staging.
- A WeCom text question produces a permission-filtered RAG answer.
- Invalid callbacks cannot call the RAG service.
- No enterprise secret appears in logs, debug payloads, or frontend.

---

## Phase D: Audit Log, Feedback And Admin APIs

**Problem:** 企业落地必须能追踪谁问了什么、命中了哪些文档、回答了什么、用户是否满意。当前只有请求日志，没有业务审计。

**Target outcome:** 每次问答都有审计记录；管理员可以查询；用户可以提交反馈；审计日志不泄露 API key 或企业微信密文。

### Files likely touched

- `app/storage/sqlite.py`
- `app/rag/service.py`
- `app/api/routes.py`
- `app/api/schemas.py`
- `app/web/static/*`
- Create: `app/audit/models.py`
- Create: `app/audit/repository.py`
- Create: `tests/test_audit_log.py`
- Create: `tests/test_feedback_api.py`

### Tasks

- [x] Create SQLite audit schema.
  - `qa_sessions`: request id, tenant id, user id hash, source, question hash, redacted question, answer summary, refusal reason, latency, token usage if available, created_at.
  - `qa_sources`: session id, source path, page, chunk id, document version.
  - `qa_feedback`: session id, rating, tags, comment, created_at.

- [x] Add audit repository.
  - Insert session after `/ask` and after stream completion.
  - Insert sources after answer generation.
  - Redact or truncate question/answer fields based on config.

- [x] Add admin query APIs.
  - `GET /admin/audit/sessions`.
  - `GET /admin/audit/sessions/{id}`.
  - Admin-only when auth enabled.

- [x] Add feedback API.
  - `POST /feedback`.
  - User can rate answer using `session_id`.
  - Optional frontend feedback buttons.

- [x] Add tests.
  - Successful answer writes audit session and sources.
  - Refusal writes refusal reason and empty sources.
  - Streaming completion writes one audit session, not one per token.
  - Non-admin cannot query audit.
  - Feedback must belong to same tenant.

### Verification

```bash
.venv/bin/pytest tests/test_audit_log.py tests/test_feedback_api.py tests/test_streaming.py -v
```

### Acceptance

- Every question has a traceable audit record.
- Audit APIs are tenant/admin protected.
- Feedback can be joined back to answer and sources.

---

## Phase E: Async Ingestion, Index Version And Rollback

**Problem:** 当前 `/ingest` 同步重建索引，生产中会阻塞请求，也无法安全回滚。

**Target outcome:** 文档入库走任务表和 worker；每次构建生成新的 `index_version`；构建成功后原子切换；失败不影响上一版索引。

### Files likely touched

- `app/core/config.py`
- `.env.example`
- `app/api/routes.py`
- `app/ingestion/pipeline.py`
- `app/rag/vector_store.py`
- `app/rag/cache.py`
- `scripts/ingest.py`
- `docker-compose.yml`
- Create: `app/ingestion/jobs.py`
- Create: `app/ingestion/index_versions.py`
- Create: `scripts/worker.py`
- Create: `tests/test_ingestion_jobs.py`
- Create: `tests/test_index_versions.py`

### Tasks

- [x] Add index settings.
  - `INDEX_ROOT_DIR=data/indexes`.
  - `ACTIVE_INDEX_VERSION_PATH=data/indexes/active_version.txt`.
  - `INGESTION_MODE=sync|async`, default `sync` for local dev.
  - `DOCUMENT_INDEX_VERSION` should be read from active version when present.

- [x] Create ingestion job schema.
  - Job fields: id, tenant id, requested_by, status, input path, target index version, error, started_at, finished_at.
  - Status values: `queued`, `running`, `succeeded`, `failed`, `cancelled`.

- [x] Add admin APIs.
  - `POST /admin/ingestion/jobs`.
  - `GET /admin/ingestion/jobs/{id}`.
  - `POST /admin/indexes/{version}/activate`.
  - `GET /admin/indexes`.

- [x] Implement worker.
  - Poll queued jobs.
  - Build Chroma/BM25/parent corpus into versioned directory.
  - Validate chunk counts and corpus status.
  - Mark job failed without touching active version on error.
  - Activate only after successful build.

- [x] Update retrieval builders.
  - Read active version.
  - Build vector store, BM25 path, and parent corpus path from active index version.
  - Cache key includes active index version.

- [x] Add rollback.
  - Admin can activate a previous successful version.
  - Keep old version directories until retention cleanup.

- [x] Add tests.
  - Failed job does not change active index.
  - Successful job creates versioned artifacts.
  - Activating previous version changes retrieval paths.
  - Cache key changes when index version changes.

### Verification

```bash
.venv/bin/pytest tests/test_ingestion_jobs.py tests/test_index_versions.py tests/test_cache.py tests/test_api_retriever_factory.py -v
```

### Acceptance

- Ingestion can run without blocking live `/ask` traffic.
- Active index can be switched and rolled back.
- Existing `make index` / `scripts.ingest` still work in local sync mode.

---

## Phase F: Production Observability, Safety And Deployment

**Problem:** 当前有基础请求日志和本地 benchmark，但还缺企业试点需要的结构化日志、指标、错误处理、部署文档和安全检查。

**Target outcome:** 系统能在 staging/企业试点环境中运行并被观测，失败时可解释、可定位、可回滚。

### Files likely touched

- `app/middleware.py`
- `app/api/routes.py`
- `app/rag/llm.py`
- `app/rag/service.py`
- `docs/deployment.md`
- `docs/production-readiness.md`
- `docs/architecture.md`
- `README.md`
- `docker-compose.yml`
- Create: `app/observability/logging.py`
- Create: `app/observability/metrics.py`
- Create: `tests/test_error_handling.py`
- Create: `tests/test_metrics.py`

### Tasks

- [x] Add unified error responses.
  - Convert LLM timeout/upstream error to safe API error.
  - For retrieval success but LLM failure, optionally return top sources with an unavailable message.
  - Do not expose stack traces to WeCom or frontend.

- [x] Add structured JSON logging.
  - Include request id, tenant id, user id hash, path, status, duration, index version.
  - Exclude API keys, WeCom secrets, encrypted payloads, full raw documents.

- [x] Add metrics endpoint or Prometheus-compatible counters.
  - Requests, errors, refusals, empty retrievals, cache hits, retrieval latency, reranker latency, LLM latency, ingestion job status.

- [x] Add rate limiting plan or implementation.
  - Minimum: document deployment-level Nginx/API Gateway rate limits.
  - Better: add per-user in-app limiter for `/ask` and WeCom callback.

- [x] Add deployment docs.
  - Required environment variables.
  - WeCom callback URL setup.
  - Secret rotation guidance.
  - Index version directories and backup.
  - Admin bootstrap.
  - Smoke test checklist.

- [x] Add security checklist.
  - Auth enabled in staging/prod.
  - Admin endpoints not internet-public.
  - Callback signature validation enabled.
  - ACL tests passing.
  - Audit retention configured.
  - LLM provider data policy recorded.

- [x] Add tests.
  - LLM timeout returns safe error.
  - Logs contain request id and no secrets.
  - Metrics increment for success, refusal, and error.

### Verification

```bash
.venv/bin/pytest tests/test_error_handling.py tests/test_metrics.py tests/test_auth_context.py tests/test_permission_retrieval.py -v
.venv/bin/python -m scripts.evaluate --variant hybrid-rerank-parent --top-k 5
```

### Acceptance

- Staging runbook exists and is executable.
- Production readiness docs clearly separate implemented features from future recommendations.
- Security and ACL tests are part of the normal verification set.

---

## Phase G: End-To-End Enterprise Smoke Scenario

**Target outcome:** Demonstrate the enterprise flow without relying on manual claims.

### Scenario

- Tenant: `default`.
- Users:
  - `alice`: department `finance`.
  - `bob`: department `hr`.
  - `admin`: role `admin`.
- Documents:
  - Finance document allowed to `finance`.
  - HR document allowed to `hr`.
  - Public policy document allowed to tenant public users.

### Required behavior

- [x] Alice can ask finance question and sees finance source.
- [x] Bob asking same finance question gets refusal or public-only answer, never finance source.
- [x] Both can ask public policy question.
- [x] WeCom callback for Alice maps to Alice context and returns same permission-filtered result.
- [x] Admin can run ingestion job and view audit.
- [x] Non-admin cannot run ingestion or view audit.
- [x] Audit log records all questions with tenant/user hash/source/index version.
- [x] Rebuilding a new index version changes cache key and keeps rollback available.

### Verification

```bash
.venv/bin/pytest tests/test_enterprise_smoke.py -v
```

If browser verification is added later:

```bash
make dev
# Open http://127.0.0.1:8000/ and verify admin/user flows manually or with Playwright.
```

### Acceptance

- One command proves the core enterprise permissions and WeCom path.
- The demo can be explained as a controlled enterprise pilot, not a production SLA claim.

---

## Documentation Updates

- [x] Update `README.md`.
  - Add enterprise mode quick start.
  - Keep local no-auth quick start.
  - Add WeCom env example without real secrets.

- [x] Update `docs/architecture.md`.
  - Add identity, ACL, audit, WeCom and async ingestion diagrams.

- [x] Update `docs/production-readiness.md`.
  - Move implemented enterprise features from "建议" to "当前已落地" only after tests pass.
  - Keep unimplemented items clearly marked as future work.

- [x] Add `docs/wecom-integration.md`.
  - Setup steps in enterprise WeChat console.
  - Callback URL.
  - Required env vars.
  - Local tunnel/staging testing notes.
  - Troubleshooting: signature mismatch, token expired, unknown user, permission denied.

- [x] Update `docs/interview-guide.md`.
  - Explain why enterprise landing is mostly identity/ACL/audit/ops, not just RAG algorithms.

---

## Suggested Commit Sequence

1. `feat: add enterprise request context and acl checks`
2. `feat: enforce document acl during retrieval`
3. `feat: add enterprise wechat integration`
4. `feat: record qa audit logs and feedback`
5. `feat: add async ingestion jobs and index versions`
6. `chore: add production observability and enterprise docs`
7. `test: add enterprise end-to-end smoke coverage`

Keep commits reviewable. Do not mix WeCom crypto, ACL filtering, and ingestion versioning in the same commit.

---

## Final Verification Before Claiming Complete

Run the focused enterprise suite:

```bash
.venv/bin/pytest \
  tests/test_auth_context.py \
  tests/test_acl.py \
  tests/test_permission_ingestion.py \
  tests/test_permission_retrieval.py \
  tests/test_wecom_signature.py \
  tests/test_wecom_routes.py \
  tests/test_wecom_handlers.py \
  tests/test_audit_log.py \
  tests/test_feedback_api.py \
  tests/test_ingestion_jobs.py \
  tests/test_index_versions.py \
  tests/test_enterprise_smoke.py \
  -v
```

Run the existing regression suite:

```bash
.venv/bin/pytest -v
node --test tests/js/ui-utils.test.mjs
```

Run quality evaluation after index changes:

```bash
.venv/bin/python -m scripts.evaluate --variant hybrid-rerank-parent --top-k 5
.venv/bin/python -m scripts.evaluate_answers --limit 10
```

Manual/staging smoke:

```bash
AUTH_ENABLED=true WECOM_ENABLED=true make dev
curl -fsS http://127.0.0.1:8000/health
```

Do not claim enterprise production readiness unless:

- Auth is enabled outside local dev.
- ACL tests pass.
- WeCom callback verification rejects invalid signatures.
- Audit logs exist for `/ask` and WeCom questions.
- Active index version is visible in logs/debug/cache keys.
- Deployment docs identify all required secrets and setup steps.
