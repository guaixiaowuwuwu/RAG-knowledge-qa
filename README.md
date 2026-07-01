# RAG 知识问答系统

一个本地可运行的企业知识库 RAG 服务。系统支持文档入库、文本清洗、表格处理、分块、向量索引、BM25 语料、混合检索、RRF 融合、BGE reranker 精排、父子块上下文展开、Query Rewrite/HyDE、LLM 生成、SSE 流式回答、引用溯源和离线评估。

当前默认实现面向本地开发和实验复现。性能、质量和生产能力结论应以仓库内的脚本、数据集、报告或实际部署记录为准，不应直接写入未经验证的吞吐、延迟或百分比提升。

完整架构见 [docs/architecture.md](docs/architecture.md)。

## 功能概览

- 文档格式：PDF、DOCX、HTML/HTM、Markdown、TXT。
- 文档理解：基础清洗、页码 metadata、标题保留、DOCX/HTML 表格转 Markdown、可选 PDF 表格抽取。
- 分块策略：child chunks 用于召回，parent chunks 用于回答上下文；表格 chunk 尽量保持行和表头完整。
- 向量检索：默认本地 `BAAI/bge-m3` embedding，Chroma 持久化到 `data/chroma`。
- 关键词检索：BM25 JSONL 语料，适合制度名、财务科目、编号和精确词。
- 融合排序：dense + BM25 候选通过 RRF 融合，再由 `BAAI/bge-reranker-v2-m3` 精排。
- 查询扩展：支持 Query Rewrite 和 HyDE，失败时回退原始问题。
- 问答接口：FastAPI `/ask` JSON 响应和 `/ask/stream` SSE 流式响应。
- 引用溯源：返回 source、page、chunk、content type、table index 等信息。
- 调试信息：`debug=true` 时返回 query variants、dense/BM25 候选、RRF 分数、reranker 分数、parent hydration 和耗时。
- 评估：检索多变体对比、答案级 RAGAS 评估、本地确定性诊断指标。
- 前端：内置静态页面，支持健康检查、重建索引、运行评估、流式问答、引用查看和 corpus status。
- 企业试点：身份上下文、管理员鉴权、文档 ACL、企业微信入口、审计/反馈、异步入库、索引版本和回滚。
- 观测与安全：结构化 JSON 请求日志、Prometheus 文本 `/metrics`、LLM 失败安全降级、本地延迟 benchmark。

## 目录结构

```text
app/
  api/                FastAPI 路由和响应 schema
  core/               配置
  evaluation/         数据集校验、检索指标、答案级评估
  ingestion/          loader、清洗、表格抽取、chunker、入库 pipeline
  integrations/       企业微信集成
  audit/              问答审计和反馈
  observability/      结构化日志和指标
  rag/                embedding、vector store、BM25、RRF、reranker、RAG service
  security/           请求上下文、鉴权和 ACL
  web/static/         本地演示页面
data/
  documents/          默认文档目录
  eval/               JSONL 评测集
  chroma/             Chroma、BM25、parent corpus 持久化目录
docs/                 架构、部署、生产化和技术取舍说明
reports/              评估和 benchmark 报告
scripts/              入库、评估、下载数据、预热、benchmark 脚本
tests/                Python 和 JS 测试
```

## 环境要求

- Python 3.11+
- Node.js 仅用于运行前端工具函数测试
- 可用的 OpenAI-compatible Chat API，用于最终回答、Query Rewrite/HyDE 和 RAGAS judge
- 首次使用本地 `bge-m3` 和 `bge-reranker-v2-m3` 时需要下载模型权重

## 快速开始

推荐使用一键入口。第一次运行先创建 `.env` 并填入 API Key：

```bash
cp .env.example .env
# 编辑 .env，至少填写 OPENAI_API_KEY
make up
```

`make up` 会自动创建 `.venv`、安装依赖、在索引缺失时运行入库并启动服务。后续日常开发通常只需要：

```bash
make dev
```

如果 `8000` 端口被占用：

```bash
PORT=8001 make dev
```

也可以拆开执行：

```bash
make setup
make index
make warmup
make dev
```

手动方式如下：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

编辑 `.env`，至少配置：

```dotenv
OPENAI_API_KEY=replace-with-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=bge-m3
```

如果使用 DeepSeek 等 OpenAI-compatible 服务，可以改成：

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-pro
EMBEDDING_MODEL=bge-m3
```

可选依赖：

```bash
# PDF 表格抽取
make install-pdf

# Redis 缓存
pip install -e ".[prod]"
```

## 配置

常用配置项见 `.env.example`：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `DOCUMENTS_DIR` | `data/documents` | 入库文档目录 |
| `CHROMA_DIR` | `data/chroma` | Chroma 持久化目录 |
| `CHROMA_COLLECTION` | `rag_knowledge_base` | Chroma collection 名称 |
| `EMBEDDING_MODEL` | `bge-m3` | 默认本地 embedding |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | reranker 模型 |
| `RETRIEVAL_TOP_K` | `4` | 问答默认返回片段数 |
| `DENSE_RETRIEVAL_TOP_K` | `20` | dense 候选数 |
| `BM25_RETRIEVAL_TOP_K` | `20` | BM25 候选数 |
| `RRF_K` | `60` | RRF 参数 |
| `QUERY_REWRITE_ENABLED` | `true` | 是否启用 query rewrite |
| `HYDE_ENABLED` | `true` | 是否启用 HyDE |
| `MAX_QUERY_VARIANTS` | `4` | 最大查询变体数 |
| `MAX_QUESTION_CHARS` | `2000` | 问题长度限制 |
| `LLM_TIMEOUT_SECONDS` | `60` | Chat API 超时 |
| `ANSWER_CACHE_ENABLED` | `false` | 是否启用答案缓存 |
| `AUTH_ENABLED` | `false` | 本地默认关闭；企业/staging 应开启 |
| `ADMIN_API_KEYS` | 空 | 管理员 bootstrap key |
| `AUDIT_DB_PATH` | `data/runtime/audit.sqlite3` | 审计和任务表 SQLite |
| `INDEX_ROOT_DIR` | `data/indexes` | 版本化索引目录 |
| `INGESTION_MODE` | `sync` | 本地同步；可设为 `async` 由 worker 入库 |
| `WECOM_ENABLED` | `false` | 企业微信入口开关 |

## 企业模式 Quick Start

本地演示默认 `AUTH_ENABLED=false`，所有接口走 local dev context，便于直接打开前端、运行 `/ask`、`/ingest` 和评估接口。企业/staging 试点必须显式开启鉴权，并准备文档 ACL manifest、管理员 API key、审计库和索引版本目录：

```dotenv
AUTH_ENABLED=true
ADMIN_API_KEYS=replace-with-random-admin-key
DEFAULT_TENANT_ID=default
PERMISSION_VERSION=staging-perm-v1
DOCUMENTS_MANIFEST_PATH=data/documents_manifest.json
AUDIT_DB_PATH=data/runtime/audit.sqlite3
INDEX_ROOT_DIR=data/indexes
ACTIVE_INDEX_VERSION_PATH=data/indexes/active_version.txt
INGESTION_MODE=async
WECOM_ENABLED=false
```

文档权限示例 `data/documents_manifest.json`：

```json
{
  "defaults": {
    "tenant_id": "default",
    "is_public": true,
    "document_version": "doc-v1"
  },
  "documents": [
    {
      "pattern": "finance/**",
      "tenant_id": "default",
      "is_public": false,
      "allowed_department_ids": ["finance"],
      "document_version": "finance-v1"
    }
  ]
}
```

管理员接口可以使用 `x-admin-api-key`、`x-api-key` 或 `Authorization: Bearer`。示例：

```bash
curl -H "x-admin-api-key: $ADMIN_API_KEY" http://127.0.0.1:8000/admin/indexes
curl -H "x-admin-api-key: $ADMIN_API_KEY" http://127.0.0.1:8000/admin/audit/sessions
```

异步入库试点：

```bash
AUTH_ENABLED=true INGESTION_MODE=async make dev
.venv/bin/python -m scripts.worker
curl -X POST http://127.0.0.1:8000/admin/ingestion/jobs \
  -H "x-admin-api-key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input_path":"data/documents","target_index_version":"index-staging-v1"}'
```

启用企业微信时补充以下变量，具体后台配置见 [docs/wecom-integration.md](docs/wecom-integration.md)：

```dotenv
WECOM_ENABLED=true
WECOM_CORP_ID=replace-with-corp-id
WECOM_AGENT_ID=replace-with-agent-id
WECOM_SECRET=replace-with-secret
WECOM_TOKEN=replace-with-token
WECOM_ENCODING_AES_KEY=replace-with-43-char-aes-key
WECOM_CALLBACK_PATH=/integrations/wecom/callback
WECOM_RESPONSE_MODE=active
```

## 数据

仓库包含两类可直接使用的公开语料：

- `data/documents/byd_chinese/`：比亚迪中文公开文档，包括年度报告、制度文件、可持续发展报告和人权政策声明等。
- `data/documents/sec_filings/`：Apple、Microsoft、NVIDIA、Amazon、Alphabet 的 SEC 10-K HTML 文件。

来源清单：

- `data/byd_chinese_manifest.json`
- `data/sec_filings_manifest.json`

也可以从 SEC EDGAR 重新下载公司年报。SEC 要求自动化请求声明 User-Agent：

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
python -m scripts.download_sec_filings
```

指定公司和数量：

```bash
python -m scripts.download_sec_filings --tickers AAPL MSFT NVDA --form 10-K --limit 2
```

## 建立索引

索引默认读取 `DOCUMENTS_DIR=data/documents`：

```bash
make index
```

等价的手动命令：

```bash
python -m scripts.ingest
```

只索引比亚迪中文语料：

```bash
DOCUMENTS_DIR=data/documents/byd_chinese python -m scripts.ingest
```

入库会生成：

- Chroma 向量索引：`data/chroma`
- BM25 语料：`data/chroma/bm25_corpus.jsonl`
- Parent corpus：`data/chroma/parent_corpus.jsonl`

本地模型首次加载较慢。需要提前预热时运行：

```bash
make warmup
```

等价的手动命令：

```bash
python -m scripts.warmup
```

## 启动服务

```bash
make dev
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

页面支持健康检查、重建索引、查看 corpus status、运行评估、流式提问和查看引用。

如果 `8000` 端口被占用：

```bash
PORT=8001 make dev
```

等价的手动命令：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

## API 示例

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

查看语料状态：

```bash
curl http://127.0.0.1:8000/corpus/status
```

查看 Prometheus 文本指标：

```bash
curl http://127.0.0.1:8000/metrics
```

普通问答：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"比亚迪信息披露事务管理制度主要规定了什么？","top_k":4}'
```

返回调试信息：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"比亚迪募集资金管理制度如何要求专户存储？","top_k":4,"debug":true}'
```

流式问答：

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"What risk does NVIDIA disclose about suppliers or manufacturing partners?","top_k":4}'
```

运行入库：

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

查看评估报告：

```bash
curl http://127.0.0.1:8000/evaluation/report
curl http://127.0.0.1:8000/evaluation/comparison-report
curl http://127.0.0.1:8000/evaluation/answer-report
```

## 可尝试的问题

中文语料：

```text
比亚迪 2024 年年度报告披露的营业额和同比增幅是多少？
比亚迪募集资金管理制度如何要求募集资金专户存储？
比亚迪人权政策声明覆盖哪些劳动者权益？
比亚迪信息披露违规时制度规定可能采取哪些处理措施？
```

英文 SEC 语料：

```text
What risk does NVIDIA disclose about suppliers or manufacturing partners?
How does Microsoft describe its cloud-related offerings?
Compare how Apple and Microsoft organize reportable segments.
```

负例：

```text
比亚迪今天的股票收盘价是多少？
What was OpenAI's audited net income in fiscal 2025?
```

## 检索评估

默认评测集是 `data/eval/sample_eval.jsonl`，包含 60 条标注样本，覆盖中文 BYD 文档、英文 SEC 10-K、事实问答、摘要、制度条款、跨文档对比和负例。字段包括 `id`、`question`、`ground_truth`、`expected_sources`、`expected_answer_keywords`，负例使用 `is_negative=true` 且 `expected_sources=[]`。

校验数据集：

```bash
python -m scripts.evaluate --validate-dataset --check-source-files
```

运行单个变体：

```bash
python -m scripts.evaluate --variant dense --top-k 5
python -m scripts.evaluate --variant hybrid --top-k 5
python -m scripts.evaluate --variant hybrid-rerank --top-k 5
python -m scripts.evaluate --variant hybrid-rerank-parent --top-k 5
python -m scripts.evaluate --variant full --top-k 5
```

运行对比：

```bash
python -m scripts.evaluate --compare --top-k 5
```

报告保存到 `reports/retrieval-*.json` 或 `reports/retrieval-comparison-*.json`。报告包含配置快照、case 数、正负例数量、逐 case retrieved sources 和 summary metrics。

当前 runtime-aligned 参考报告：

- `reports/retrieval-20260623-185716.json`
- `reports/retrieval-comparison-20260623-190622.json`

这两份报告的配置快照记录 `versioned_indexing_enabled=false`，effective index artifacts 指向默认本地路径 `data/chroma`、`data/chroma/bm25_corpus.jsonl` 和 `data/chroma/parent_corpus.jsonl`。

指标：

- `hit_rate_at_k`：正例 Top-K 中是否命中任一标注来源。
- `mrr_at_k`：第一个命中来源的倒数排名均值。
- `source_recall_at_k`：标注来源被 Top-K 找回的比例。
- `precision_at_k`：Top-K 来源中属于标注来源的比例。
- `ndcg_at_k`：考虑命中排名位置的归一化折损累计增益。
- `negative_rejection_rate`：负例问题没有返回检索结果的比例。

## 答案级评估

答案级评估会调用真实 RAG 链路，保存问题、标准答案、生成答案、检索上下文、引用来源、模型配置、本地诊断指标和 RAGAS 指标。

```bash
python -m scripts.evaluate_answers --limit 10
```

RAGAS 数据集字段为 `question`、`answer`、`contexts`、`ground_truth`。当前指标包括：

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

RAGAS 会复用 `.env` 中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `CHAT_MODEL` 作为 judge LLM；embedding 固定使用本地 `bge-m3`。缺少依赖、凭据或模型调用失败时，命令会非零退出，并保存 `reports/answer-eval-failed-*.json` 部分报告。

当前 runtime-aligned 参考报告：`reports/answer-eval-20260623-191226.json`。旧的 `reports/answer-eval-20260623-134526.json` 是索引路径错配期间生成的过期零分报告，不能作为当前质量证据引用。

## Benchmark

本地延迟 smoke benchmark：

```bash
python -m scripts.benchmark --limit 20 --debug
```

报告保存到 `reports/latency-benchmark-*.json`。该 benchmark 是单进程串行本地测试，结果受机器、模型缓存、语料规模和网络影响，不能直接作为生产吞吐或 SLA 证明。

## Docker

本地容器运行：

```bash
make docker-up
```

等价的手动命令：

```bash
docker compose up --build api
curl http://127.0.0.1:8000/health
```

启用 Redis profile：

```bash
docker compose --profile redis up --build api redis
```

在 `.env` 中开启缓存：

```dotenv
ANSWER_CACHE_ENABLED=true
ANSWER_CACHE_BACKEND=redis
ANSWER_CACHE_TTL_SECONDS=300
REDIS_URL=redis://redis:6379/0
```

缓存只用于普通 `/ask` 的最终答案和引用，不缓存 `/ask/stream`。

部署和生产化说明：

- [docs/deployment.md](docs/deployment.md)
- [docs/production-readiness.md](docs/production-readiness.md)
- [docs/tradeoffs.md](docs/tradeoffs.md)

## 测试

```bash
pytest -q
node --test tests/js/ui-utils.test.mjs
```

也可以使用虚拟环境中的解释器：

```bash
.venv/bin/pytest -q
```

## 常见问题

### 首次运行很慢

默认 embedding 和 reranker 都是本地模型，首次运行会下载并加载 `BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3`。演示或评估前可以先运行：

```bash
python -m scripts.warmup
```

### 问答接口报 OPENAI_API_KEY 错误

最终回答、Query Rewrite/HyDE 和 RAGAS judge 都需要 OpenAI-compatible Chat API。请检查 `.env` 中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `CHAT_MODEL`。

### `/ask` 没有引用来源

先检查索引是否已经构建：

```bash
curl http://127.0.0.1:8000/corpus/status
```

如果 BM25、parent corpus 或 Chroma 为空，重新入库：

```bash
python -m scripts.ingest
```

### PDF 表格没有被抽取

PDF 表格抽取依赖可选依赖：

```bash
pip install -e ".[pdf-tables]"
```

扫描版 PDF/OCR 不是默认入库硬依赖，需要额外 OCR/版面分析能力。

## 已知边界

- 默认 Chroma、BM25 JSONL 和 parent JSONL 适合本地开发，不等同于生产级向量检索和关键词检索服务。
- Redis cache 默认关闭；cache key 已纳入 tenant、user hash、permission version 和 active index version。
- 负例拒答已包含空检索、低置信、时间敏感和私有/不可用问题策略，但仍需要真实企业语料评估调阈值。
- 复杂跨页表格、图片和扫描 PDF 需要额外文档理解能力。
- 当前 `/metrics` 是进程内指标，SQLite 用于审计和任务表；多副本生产应接 Prometheus/OTel 和企业数据库。
- 生产环境仍需要入口层限流、集中告警、压测、备份恢复演练和质量监控。
