# 生产可用性说明

本文档记录当前仓库已经落地的生产化脚手架、仍属于本地开发版的部分，以及面向生产的改造路径。它的目的不是制造生产证明，而是让项目在面试叙述中边界清晰、可验证、可扩展。

## 当前已落地

- FastAPI 服务和静态前端。
- Dockerfile 与 Docker Compose 本地栈。
- 可选 Redis 答案缓存抽象，默认关闭。
- `/ask` 和 `/ask/stream` 的问题长度限制：`MAX_QUESTION_CHARS`。
- Chat LLM 客户端请求超时：`LLM_TIMEOUT_SECONDS`。
- Query Rewrite/HyDE 超时：`QUERY_TRANSFORM_TIMEOUT_SECONDS`。
- 企业模式身份上下文：`tenant_id`、`user_id`、部门、角色、权限版本和来源。
- 文档 ACL 贯穿入库 metadata、Chroma/BM25/parent hydration、RAG service 和答案缓存 key。
- 企业微信回调验签、文本消息处理、用户映射和主动/被动回复。
- 问答审计、引用记录、反馈 API 和管理员审计查询。
- 异步入库任务、版本化索引目录、active index 原子切换和回滚入口。
- 请求级结构化 JSON 日志：request id、tenant id、user id hash、path、status、duration、index version。
- Prometheus 文本指标端点 `/metrics`：请求量、错误量、拒答、空检索、retrieval latency、LLM latency、入库任务状态。
- LLM 失败安全降级：最终生成服务不可用时返回安全提示，并可附带已通过权限过滤的来源。
- Debug trace 中包含检索和 LLM 粗粒度耗时；不会返回 API key、企业微信密文或原始秘钥。
- 本地延迟 benchmark：`python -m scripts.benchmark --limit 20 --debug`。

## 配置项

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `MAX_QUESTION_CHARS` | `2000` | 限制单次问题长度，降低 prompt 注入、超长输入和成本风险 |
| `LLM_TIMEOUT_SECONDS` | `60` | OpenAI-compatible chat client 超时 |
| `ANSWER_CACHE_ENABLED` | `false` | 是否启用最终答案缓存 |
| `ANSWER_CACHE_BACKEND` | `redis` | 缓存后端，可选 `redis` 或测试用 `memory` |
| `ANSWER_CACHE_TTL_SECONDS` | `300` | 答案缓存 TTL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 连接地址 |
| `AUTH_ENABLED` | `false` | 本地默认关闭；staging/prod 必须开启 |
| `ADMIN_API_KEYS` | 空 | 管理员 bootstrap key，生产应由密钥系统注入 |
| `AUDIT_DB_PATH` | `data/runtime/audit.sqlite3` | 审计、反馈和入库任务 SQLite 路径 |
| `INDEX_ROOT_DIR` | `data/indexes` | 版本化索引根目录 |
| `ACTIVE_INDEX_VERSION_PATH` | `data/indexes/active_version.txt` | 当前 active index 指针 |
| `INGESTION_MODE` | `sync` | 本地同步；试点建议使用 `async` + worker |
| `WECOM_ENABLED` | `false` | 企业微信入口开关 |

## Redis 缓存设计

缓存对象：

- 缓存普通 `/ask` 的最终 `answer` 和 `sources`。
- 不缓存 `/ask/stream`，避免缓存半截 token 流或 SSE 事件顺序。
- 不缓存入库、评估和 corpus status。

缓存 key：

- 问题文本去首尾空白。
- `top_k`。
- `chat_model`。
- `embedding_model`。
- `chroma_collection`。
- 检索选项：rewrite、HyDE、parent hydration、reranker、max variants。

失效策略：

- TTL 默认 300 秒。
- 缓存 key 已包含 active index version、tenant、user hash 和 permission version，避免跨用户或索引版本复用。
- Redis 不可用时会记录 warning 并降级为无缓存，不阻断问答。

适合缓存的场景：

- 高频重复问题。
- 演示或内部知识库中的固定 FAQ。
- Reranker/LLM 成本较高且答案时效性要求不高的问题。

不适合缓存的场景：

- 权限高度个性化的问题，除非 key 包含用户和权限版本。
- 文档频繁更新且没有 index version 的场景。
- 需要每次实时生成或审计 token 流的场景。

## 请求限制与超时

当前限制：

- `top_k` 由 Pydantic schema 限制在 `1..20`。
- `MAX_QUESTION_CHARS` 默认限制问题长度为 2000 字符。
- `QUERY_TRANSFORM_TIMEOUT_SECONDS` 限制 Query Rewrite/HyDE 调用。
- `LLM_TIMEOUT_SECONDS` 限制 chat completion 调用。

当前未内置 per-user 令牌桶限流；试点部署必须在入口层配置限流，建议基线：

- Nginx/API Gateway 对 `/ask` 和企业微信 callback 设置每用户或每 IP 限流，例如 60 req/min，burst 10。
- 对 `/admin/*` 设置更严格限流，例如 10 req/min，并限制来源网段。
- 设置请求体大小上限，企业微信 callback 和 `/ask` 不应接收大文件正文。

后续可增强：

- 增加应用内 per-user limiter，对 `RequestContext.user_id` + `tenant_id` 计数。
- 为 embedding、reranker、LLM 分别设置重试、熔断和超时预算。

## 降级策略

### LLM API 不可用

当前行为：

- 普通问答不会向前端或企业微信暴露上游堆栈。
- 检索成功但最终 LLM 失败时，返回“回答服务暂时不可用，请稍后重试。”，并保留已通过 ACL 过滤的 sources。
- 审计记录 `refusal_reason=llm_unavailable`，指标记录 `rag_errors_total{stage="llm"}`。
- Query Rewrite/HyDE 的失败会被 `QueryTransformer` 捕获，回退到原始问题。

后续可增强：

- 为不同模型提供路由降级，例如主模型失败后切到备用模型。

### Reranker 不可用

当前行为：

- 默认构建 BGE reranker，缺少依赖或模型加载失败会显式报错。
- benchmark/debug 路径支持通过 `RetrievalDebugOptions.reranker_enabled=False` 关闭 reranker。

生产建议：

- 将 reranker 服务化并设置超时。
- 超时或失败时返回 RRF 排序结果，并在 trace/log 中标记 degraded。
- 保留评估报告，量化关闭 reranker 对召回和排序的影响。

### 空检索结果

当前行为：

- 返回“知识库中没有找到相关内容，无法基于现有资料回答。”
- sources 为空。

生产建议：

- 展示可操作建议：换关键词、确认文档已入库、联系管理员。
- 记录空检索 query，用于补充语料和优化 query expansion。

### 索引未构建

当前行为：

- BM25 corpus 缺失时返回空 BM25。
- Chroma collection 为空时 dense 结果为空。
- `/corpus/status` 可查看文档数、chunk 数、BM25、parent corpus 和 Chroma 状态，并通过 `ready` / `readiness_reason` 明确区分可用索引与空索引。
- 当前已验证的本地默认索引为 `data/chroma`，`ready=true`、Chroma chunks 5171、BM25 rows 5171、parent rows 2309。
- 异步入库 worker 构建新版本目录，校验成功后才切换 active index；失败不会改变 active index。
- 管理员可通过 `/admin/indexes/{version}/activate` 回滚到上一版成功索引。

生产建议：

- 启动时做 readiness check，索引未就绪则不接流量。
- 为 `data/indexes` 和 `data/runtime/audit.sqlite3` 做备份与保留策略。

## 可观测性

当前日志：

- `app.requests` logger 输出 JSON 字段：`request_id`、HTTP method、path、status、duration、tenant id、user id hash、active index version。
- 响应头包含 `x-request-id` 和 `x-process-time-ms`。
- Debug trace 包含 query variants、dense candidates、BM25 candidates、RRF scores、reranker scores、parent hydration、final chunks 和粗粒度 timings。

当前指标：

- `/metrics` 暴露 Prometheus 文本格式。
- `rag_http_requests_total`、`rag_http_errors_total`、`rag_http_request_duration_ms_*`。
- `rag_refusals_total`、`rag_empty_retrievals_total`、`rag_errors_total`。
- `rag_retrieval_latency_ms_*`、`rag_llm_latency_ms_*`。
- `rag_ingestion_jobs_total`。

生产建议：

- 日志采集侧补充 environment、service version、pod/host 等部署字段。
- 增加 cache hit rate、reranker latency、token usage 和 worker 循环指标。
- Trace：将 retrieval、rerank、prompt assembly、LLM call 拆成 span。
- 质量监控：定期运行 `scripts.evaluate` 和 `scripts.evaluate_answers`，保存趋势。
- 可选工具：LangSmith、Phoenix、OpenTelemetry、Prometheus/Grafana、Sentry。

## 安全检查清单

staging/prod 上线前必须逐项确认：

- `AUTH_ENABLED=true`，本地 fallback 只允许开发环境使用。
- 管理员接口 `/admin/*`、`/ingest`、`/evaluation/*`、`/corpus/status` 不直接暴露到公网，且需要管理员凭据。
- `ADMIN_API_KEYS`、`OPENAI_API_KEY`、`WECOM_SECRET`、`WECOM_TOKEN`、`WECOM_ENCODING_AES_KEY` 由密钥系统注入，不写入 Git。
- 企业微信 callback 使用官方验签；无效签名不能调用 RAG。
- 文档 manifest 中为非公开文档配置 tenant、部门、用户或角色 ACL。
- ACL 测试、权限检索测试、企业微信签名测试、审计测试在发布前通过。
- 审计库配置备份、保留期和访问权限；日志中只记录 user hash，不记录原始 API key 或密文。
- LLM provider 数据使用政策已记录并经企业侧确认。
- `data/indexes` 有备份与回滚演练，active index 切换可追溯。
- 发布前至少运行企业 smoke：`tests/test_enterprise_smoke.py`，确认 Web/API、企业微信、ACL、审计、异步入库和回滚在同一场景中连通。

## 本地延迟 Benchmark

命令：

```bash
python -m scripts.benchmark --limit 20 --debug
```

输出：

- `reports/latency-benchmark-YYYYMMDD-HHMMSS.json`
- 每个 case 的 total latency、retrieval latency、LLM latency、source count、answer chars。
- summary 中包含 mean、p50、p95、max。

限制：

- 这是单进程串行 smoke benchmark。
- 受本机 CPU/GPU、模型缓存、网络、LLM provider、语料规模影响很大。
- 不能把它当作生产 QPS 或 SLA 证明。

## 向生产演进的优先级

1. 抽象向量库 adapter，增加 Milvus/Qdrant/pgvector 实现。
2. 增加应用内 per-user rate limiter 和 cache hit rate 指标。
3. 拆分 retrieval、rerank、prompt assembly、LLM call 为 OpenTelemetry spans。
4. 增加文档删除同步、索引保留清理和灾备恢复演练。
5. 将 SQLite 审计和任务表迁移到企业数据库。
6. 扩展端到端企业 smoke 到真实 staging 数据、真实企业微信后台和浏览器自动化。
