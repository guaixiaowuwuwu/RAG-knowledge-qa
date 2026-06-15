# 生产可用性说明

本文档记录当前仓库已经落地的生产化脚手架、仍属于本地开发版的部分，以及面向生产的改造路径。它的目的不是制造生产证明，而是让项目在面试叙述中边界清晰、可验证、可扩展。

## 当前已落地

- FastAPI 服务和静态前端。
- Dockerfile 与 Docker Compose 本地栈。
- 可选 Redis 答案缓存抽象，默认关闭。
- `/ask` 和 `/ask/stream` 的问题长度限制：`MAX_QUESTION_CHARS`。
- Chat LLM 客户端请求超时：`LLM_TIMEOUT_SECONDS`。
- Query Rewrite/HyDE 超时：`QUERY_TRANSFORM_TIMEOUT_SECONDS`。
- 请求级结构化日志：method、path、status、duration、request id。
- Debug trace 中包含检索和 LLM 粗粒度耗时。
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
- 重建索引后，生产环境应增加显式 index version 或 corpus version 到 key 中；当前本地版使用 collection name 隔离，适合 demo，不足以处理多租户生产索引版本。
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

生产建议：

- 在 API Gateway/Nginx 层增加请求体大小限制。
- 增加用户级和 IP 级限流。
- 将 `/ingest` 移到鉴权后台或异步任务队列。
- 为 embedding、reranker、LLM 分别设置重试、熔断和超时预算。

## 降级策略

### LLM API 不可用

当前行为：

- 普通问答会抛出上游异常。
- Query Rewrite/HyDE 的失败会被 `QueryTransformer` 捕获，回退到原始问题。

生产建议：

- 对最终回答 LLM 加统一异常处理，返回“检索到资料但生成服务暂不可用”的错误。
- 可在失败时返回 Top-K 引用片段，避免用户完全无结果。
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
- `/corpus/status` 可查看文档数、chunk 数、BM25 和 Chroma 状态。

生产建议：

- 启动时做 readiness check，索引未就绪则不接流量。
- 索引重建使用新版本目录，完成后原子切换。
- 保留上一版可用索引用于回滚。

## 可观测性

当前日志：

- `app.requests` logger 输出 `request_id`、HTTP method、path、status code、duration。
- 响应头包含 `x-request-id` 和 `x-process-time-ms`。
- Debug trace 包含 query variants、dense candidates、BM25 candidates、RRF scores、reranker scores、parent hydration、final chunks 和粗粒度 timings。

生产建议：

- 使用 JSON 日志格式，加入 environment、service version、user id hash、tenant id、corpus version。
- 指标：请求量、错误率、空检索率、cache hit rate、retrieval latency、rerank latency、LLM latency、token usage。
- Trace：将 retrieval、rerank、prompt assembly、LLM call 拆成 span。
- 质量监控：定期运行 `scripts.evaluate` 和 `scripts.evaluate_answers`，保存趋势。
- 可选工具：LangSmith、Phoenix、OpenTelemetry、Prometheus/Grafana、Sentry。

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

1. 把 `/ingest` 从同步 API 改为队列任务，并加入索引版本。
2. 抽象向量库 adapter，增加 Milvus/Qdrant/pgvector 实现。
3. 引入 Redis cache hit rate 指标和 corpus version cache key。
4. 增加统一异常处理中间件，将上游错误转为可解释 API 响应。
5. 接入 OpenTelemetry trace 和 JSON logs。
6. 为权限过滤、文档删除、索引回滚补测试。
