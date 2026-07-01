# 部署说明

本文档说明当前仓库的本地容器化部署方式，以及向生产环境演进时需要替换或增强的组件。这里不包含任何线上吞吐、P95 或可用性承诺；相关数字必须来自实际压测报告。

## 本地 Docker 运行

准备环境文件：

```bash
cp .env.example .env
```

编辑 `.env`，至少配置：

```dotenv
OPENAI_API_KEY=replace-with-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=bge-m3
ANSWER_CACHE_ENABLED=false
```

构建并启动 API：

```bash
docker compose up --build api
```

验证健康检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

Compose 会把本机目录挂载到容器：

- `./data/documents` -> `/app/data/documents`
- `./data/chroma` -> `/app/data/chroma`
- `./data/indexes` -> `/app/data/indexes`
- `./data/runtime` -> `/app/data/runtime`
- `./reports` -> `/app/reports`

因此索引和报告会留在仓库目录中，便于本地复现实验。

## 可选 Redis

Redis 缓存默认关闭。需要验证缓存路径时启动 Redis profile：

```bash
docker compose --profile redis up --build api redis
```

在 `.env` 中打开缓存：

```dotenv
ANSWER_CACHE_ENABLED=true
ANSWER_CACHE_BACKEND=redis
ANSWER_CACHE_TTL_SECONDS=300
REDIS_URL=redis://redis:6379/0
```

缓存的是普通 `/ask` 的最终答案和引用，不缓存 `/ask/stream` 的 SSE token 流。缓存 key 由问题文本、`top_k`、chat model、embedding model、Chroma collection 和检索选项组成，避免不同配置共用旧答案。

## 企业微信入口

企业微信自建应用回调入口在 Phase C 中已接入，默认关闭。开启方式、环境变量、后台配置和测试命令见 [企业微信集成说明](wecom-integration.md)。生产或 staging 必须同时开启 `AUTH_ENABLED=true`，并确保用户映射和文档 ACL 已通过测试。

最小 staging 环境变量：

```dotenv
AUTH_ENABLED=true
ADMIN_API_KEYS=replace-with-random-admin-key
DEFAULT_TENANT_ID=default
PERMISSION_VERSION=staging-perm-v1
AUDIT_DB_PATH=data/runtime/audit.sqlite3
DOCUMENTS_MANIFEST_PATH=data/documents_manifest.json
INDEX_ROOT_DIR=data/indexes
ACTIVE_INDEX_VERSION_PATH=data/indexes/active_version.txt
INGESTION_MODE=async
WECOM_ENABLED=true
WECOM_CORP_ID=replace-with-corp-id
WECOM_AGENT_ID=replace-with-agent-id
WECOM_SECRET=replace-with-secret
WECOM_TOKEN=replace-with-token
WECOM_ENCODING_AES_KEY=replace-with-43-char-aes-key
WECOM_RESPONSE_MODE=active
```

管理员接口示例：

```bash
curl -H "x-admin-api-key: $ADMIN_API_KEY" http://127.0.0.1:8000/admin/indexes
curl -H "x-admin-api-key: $ADMIN_API_KEY" http://127.0.0.1:8000/admin/audit/sessions
```

不要把这些密钥写入仓库、镜像或前端配置。轮换时先同时接受新旧凭据，验证后移除旧凭据；企业微信后台同步更新 callback token/AES key 后，应立即用无效签名测试确认旧签名被拒绝。

## 可选向量库服务

当前代码默认使用 Chroma 本地持久化目录。Compose 提供 Qdrant profile 作为向量库服务示例：

```bash
docker compose --profile qdrant up qdrant
```

注意：当前业务代码尚未实现 Qdrant/Milvus adapter，不能把该 profile 描述为已接入生产向量库。迁移步骤应是：

1. 在 `app/rag/vector_store.py` 旁新增 `VectorStore` 协议和 Milvus/Qdrant adapter。
2. 保持 `similarity_search(query, top_k)` 和 `add_chunks(chunks)` 行为与 Chroma adapter 一致。
3. 将 collection/schema、向量维度、metric type、index type 写入配置。
4. 用 `scripts.evaluate --compare --top-k 5` 对迁移前后的召回指标做对比。
5. 再做压测和故障演练，不用本地 demo 结果替代生产结论。

Milvus 常见生产权衡：

- 优点：适合较大规模向量检索、索引类型丰富、服务化部署成熟。
- 代价：部署和运维复杂度高于本地 Chroma，需要关注 collection schema、索引构建时间、内存、水位和备份。
- 面试表述：本项目默认 Chroma 是本地可复现选择，Milvus 是生产候选方案，不是已压测事实。

## 运行索引与评估

容器内重建索引：

```bash
docker compose run --rm api python -m scripts.ingest
```

异步入库 worker：

```bash
INGESTION_MODE=async docker compose up --build api worker
curl -X POST http://127.0.0.1:8000/admin/ingestion/jobs \
  -H "x-admin-api-key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'
```

索引版本目录写入 `data/indexes/<version>/`，active 指针写入 `data/indexes/active_version.txt`。备份至少应覆盖 `data/indexes/`、`data/runtime/audit.sqlite3` 和文档 manifest。回滚示例：

```bash
curl -X POST http://127.0.0.1:8000/admin/indexes/index-previous/activate \
  -H "x-admin-api-key: $ADMIN_API_KEY"
```

运行检索对比：

```bash
docker compose run --rm api python -m scripts.evaluate --compare --top-k 5
```

运行本地延迟 smoke benchmark：

```bash
docker compose run --rm api python -m scripts.benchmark --limit 20 --debug
```

报告会写入 `reports/`。这些数字只能表述为当前机器或当前容器环境的本地结果。

## 生产部署建议

生产环境通常还需要：

- 反向代理：Nginx、Ingress 或 API Gateway，负责 TLS、请求体大小、超时和基础限流。
- 鉴权：当前支持本地管理员 API key 和企业微信用户映射；生产可继续接企业 SSO/IAM。
- 权限过滤：当前已在入库 metadata、dense、BM25、parent hydration、RAG service 和 cache key 中执行 ACL；上线前必须跑权限测试。
- 异步入库：当前已有任务表、worker、索引版本和回滚；生产仍需队列化、并发控制和任务告警。
- 观测：当前有 JSON request log、`/metrics`、错误指标和粗粒度 retrieval/LLM latency；生产仍需集中化日志、告警和 trace。
- 数据治理：当前有索引版本和审计记录；生产仍需删除同步、保留期、备份恢复和数据库迁移。

## Smoke Test Checklist

staging 部署后执行：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/metrics
.venv/bin/pytest tests/test_auth_context.py tests/test_permission_retrieval.py tests/test_wecom_signature.py -v
.venv/bin/pytest tests/test_error_handling.py tests/test_metrics.py -v
```

人工检查：

- 无管理员凭据访问 `/admin/indexes` 返回 401/403。
- 企业微信无效 callback 签名返回 403，且不调用 RAG。
- 管理员能查看 active index 和审计 session。
- 普通用户只看到自己 ACL 允许的来源。
- 日志中有 request id、tenant id、user hash、index version，且没有 API key、企业微信密文或完整原始文档。
