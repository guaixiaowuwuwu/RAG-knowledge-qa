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
```

Compose 会把本机目录挂载到容器：

- `./data/documents` -> `/app/data/documents`
- `./data/chroma` -> `/app/data/chroma`
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
- 鉴权：接入企业 SSO、API key 或服务间认证。
- 权限过滤：检索前后都要按用户权限过滤文档和 chunk。
- 异步入库：文档上传、解析、embedding 和索引构建应进入队列，避免阻塞 API worker。
- 观测：结构化日志、指标、trace、错误告警和评估报告趋势。
- 数据治理：索引版本、回滚、删除同步、备份和数据保留策略。
