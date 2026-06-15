# RAG 知识问答系统

这是一个用于复现面试项目的 RAG 知识问答系统。它支持本地文档导入、分块、向量化、Chroma + BM25 混合检索、RRF 融合、BGE reranker 精排、LLM 回答生成和引用来源返回。

## 项目定位与声明边界

本项目定位为企业知识库 RAG 的本地可演示实现，而不是已经完成生产压测的线上系统。系统能力、评估指标和面试叙述必须以仓库内可运行代码、数据集、评估脚本、报告或明确的人工验证记录为依据。

所有检索指标都应由 `python -m scripts.evaluate` 生成；答案级指标应由基于 RAGAS 的 `python -m scripts.evaluate_answers` 生成。不要直接复用模板网站中的示例数字，例如未由本仓库实验产出的百分比提升、日 QPS、P95 延迟或线上可用性承诺。

完整离线入库、在线问答链路和本地开发版/生产版边界见 `docs/architecture.md`。

## 功能范围

- 支持 `txt`、`md`、`pdf`、`docx`、`html` 文档。
- 使用递归文本分块。
- 默认使用本地 `bge-m3` embedding 模型。
- 使用 Chroma 本地向量库和 BM25 关键词检索。
- 使用 RRF 融合 dense 与 BM25 候选结果。
- 使用 `BAAI/bge-reranker-v2-m3` 作为必经 reranker 精排模型。
- 使用 FastAPI 提供索引和问答接口。
- 返回答案和引用片段，支持普通 JSON 和 SSE 流式回答。
- 文档解析会做基础清洗，DOCX/HTML 表格会转为 Markdown；PDF 表格可在安装 `pdfplumber` 后启用。
- 提供 Docker/Compose 本地部署脚手架、默认关闭的 Redis 答案缓存、请求长度限制、LLM 超时、结构化请求日志和本地延迟 benchmark。

## 第二阶段能力

- 支持 `docx`、`html`、`htm` 文档解析。
- 建立索引时会同时生成 Chroma 向量索引和 BM25 JSONL 语料。
- 问答默认走混合检索：Chroma 稠密检索 + BM25 稀疏检索 + RRF 融合。
- 混合检索候选结果会经过 BGE Reranker 精排。
- `/ask/stream` 支持 SSE 流式输出。
- `python -m scripts.evaluate` 可运行检索评估、数据集校验和多变体对比。
- `python -m scripts.evaluate_answers` 可运行基于 RAGAS 的答案级自动化评估，并保留本地确定性诊断指标。
- `python -m scripts.warmup` 可预热本地 embedding 和 reranker 模型。

## 第四阶段能力

- 建立索引时生成父块语料，检索命中子块后返回父块上下文。
- 支持 Query 改写和 HyDE 查询扩展，默认最多生成 4 个查询变体。
- 评估脚本输出 Hit Rate@K、MRR@K、Source Recall@K、Precision@K、NDCG@K 和负例拒答率。
- 前端可以直接运行评估并查看指标。

## 第八阶段能力

- 支持基础文档清洗：归一化空白、尽量去除可检测的重复页眉页脚、保留页码和标题元数据。
- DOCX 会提取段落、标题和表格；HTML 会优先抽取语义内容、去除导航噪声并保留表格。
- 表格会转换成 Markdown，分块时会尽量保持表格完整，超长表格按完整行切分并重复表头。
- 扫描版 PDF/OCR 仍是可选增强项，不作为默认入库硬依赖。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

编辑 `.env`，填入可用的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `CHAT_MODEL`。

默认 `EMBEDDING_MODEL=bge-m3`，会在本地通过 `sentence-transformers` 加载 `BAAI/bge-m3` 生成向量，不需要配置 embedding API URL。问答检索链路还会加载 `BAAI/bge-reranker-v2-m3` 进行精排。首次运行会下载模型权重，耗时取决于网络和机器性能。

如果希望启用 PDF 表格抽取的可选增强，可以额外安装：

```bash
pip install -e ".[pdf-tables]"
```

答案级自动化评估强制复用业务问答的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `CHAT_MODEL` 作为 RAGAS judge LLM；不再单独配置 `RAGAS_API_KEY`、`RAGAS_BASE_URL` 或 `RAGAS_JUDGE_MODEL`。RAGAS embedding 固定使用本地 `bge-m3`，不会调用 OpenAI embedding API。

如果使用 DeepSeek 做回答生成，可以配置：

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-pro
EMBEDDING_MODEL=bge-m3
```

## 常用命令

```bash
python -m scripts.ingest
python -m scripts.evaluate --variant full --top-k 5
python -m scripts.evaluate_answers --limit 10
python -m scripts.benchmark --limit 20 --debug
python -m scripts.warmup
uvicorn app.main:app --reload
```

## 建立索引

```bash
python -m scripts.ingest
```

## 真实公司文档数据

项目内置 `scripts.download_sec_filings`，可从 SEC EDGAR 下载公开公司年报作为真实企业语料。默认下载 Apple、Microsoft、NVIDIA、Amazon、Alphabet 的最新 `10-K`，保存到 `data/documents/sec_filings/`，并生成 `data/sec_filings_manifest.json` 记录来源 URL。

SEC 要求自动化请求声明 User-Agent，并限制请求速率。建议先配置可识别的联系信息：

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

下载默认公司年报：

```bash
python -m scripts.download_sec_filings
```

也可以指定公司、表单类型和每家公司下载数量：

```bash
python -m scripts.download_sec_filings --tickers AAPL MSFT NVDA --form 10-K --limit 2
```

下载完成后重建索引：

```bash
python -m scripts.ingest
```

## 中文公司文档数据

`data/documents/byd_chinese/` 中放入了一批比亚迪中文公开文档，包含年度报告、可持续发展报告、信息披露事务管理制度、募集资金管理制度、市值管理制度和人权政策声明等。来源清单见 `data/byd_chinese_manifest.json`。

如果希望演示中文公司制度和经营材料，建议只索引这批中文文档，速度更快且检索结果更集中：

```bash
DOCUMENTS_DIR=data/documents/byd_chinese python -m scripts.ingest
DOCUMENTS_DIR=data/documents/byd_chinese uvicorn app.main:app --reload
```

可尝试提问：

```text
比亚迪信息披露事务管理制度主要规定了什么？
比亚迪募集资金管理制度如何要求专户存储和使用？
比亚迪人权政策声明覆盖哪些劳动者权益？
比亚迪 2025 年年度报告中如何描述新能源汽车出口和全球化？
```

## 启动服务

```bash
uvicorn app.main:app --reload
```

## 前端页面

启动服务后打开：

```bash
uvicorn app.main:app --reload
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

页面支持健康检查、重建索引、运行评估、流式提问和引用来源查看。演示前建议先运行：

```bash
python -m scripts.ingest
python -m scripts.warmup
```

## 调用接口

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG 系统包含哪些核心步骤？","top_k":4}'
```

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG 系统包含哪些核心步骤？","top_k":4}'
```

```bash
curl http://127.0.0.1:8000/evaluation/report
```

如果 `8000` 端口已被占用，可以换一个端口启动服务：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001
curl http://127.0.0.1:8001/health
```

## Docker 与生产化脚手架

本地容器运行：

```bash
docker compose up --build api
curl http://127.0.0.1:8000/health
```

Redis 缓存默认关闭；需要验证缓存链路时：

```bash
docker compose --profile redis up --build api redis
```

相关文档：

- `docs/deployment.md`：Docker Compose、本地 Redis、向量库迁移说明。
- `docs/production-readiness.md`：缓存 key/TTL、请求限制、超时、降级策略、观测和 benchmark 边界。

本地延迟 benchmark：

```bash
python -m scripts.benchmark --limit 20 --debug
```

报告保存到 `reports/latency-benchmark-*.json`。这是本地串行 smoke benchmark，不能作为生产 QPS 或 SLA 证明。

## 测试

```bash
pytest
```

```bash
node --test tests/js/ui-utils.test.mjs
```

## 检索评估

```bash
python -m scripts.ingest
python -m scripts.evaluate --validate-dataset --check-source-files
python -m scripts.evaluate --variant full --top-k 5
python -m scripts.evaluate --compare --top-k 5
```

默认评测集是 `data/eval/sample_eval.jsonl`，包含 60 条标注问题：比亚迪中文年度报告和制度文件、SEC 10-K 英文问题、精确事实、摘要、跨文档对比和不在知识库中的负例。数据集字段包括 `id`、`question`、`ground_truth`、`expected_sources`、`expected_answer_keywords`，负例使用 `is_negative=true` 且 `expected_sources=[]`。

检索评估支持这些变体：

```bash
python -m scripts.evaluate --variant dense --top-k 5
python -m scripts.evaluate --variant hybrid --top-k 5
python -m scripts.evaluate --variant hybrid-rerank --top-k 5
python -m scripts.evaluate --variant hybrid-rerank-parent --top-k 5
python -m scripts.evaluate --variant full --top-k 5
```

`--compare` 会依次运行全部变体，并生成 Markdown 表格和 JSON 报告。报告默认保存到 `reports/retrieval-YYYYMMDD-HHMMSS.json` 或 `reports/retrieval-comparison-YYYYMMDD-HHMMSS.json`，其中包含数据集路径、配置快照、case 数、正负例数、逐 case retrieved sources 和 summary metrics。

指标含义：

- `hit_rate_at_k`：正例 Top-K 中是否命中任一标注来源。
- `mrr_at_k`：第一个命中来源的倒数排名均值。
- `source_recall_at_k`：标注来源被 Top-K 找回的比例。
- `precision_at_k`：Top-K 来源中属于标注来源的比例。
- `ndcg_at_k`：考虑命中排名位置的归一化折损累计增益。
- `negative_rejection_rate`：负例问题没有返回检索结果的比例。

不要手写或复用模板指标。README、简历或面试材料中的最新表格应来自 `reports/retrieval-comparison-*.json` 的真实输出。`full` 变体会启用 Query Rewrite/HyDE，通常需要可用的 `OPENAI_API_KEY`；rerank 变体会加载 `BAAI/bge-reranker-v2-m3`；所有本地 embedding 默认使用 `BAAI/bge-m3`。

## 答案级评估

```bash
python -m scripts.evaluate_answers --limit 10
```

答案级评估会走真实 `RagService`，保存 `question`、`ground_truth`、生成答案、检索上下文、引用来源、模型配置、本地确定性诊断指标和 RAGAS 指标。RAGAS 是必需评估路径；如果答案生成失败、缺少 `ragas`/`datasets` 依赖、缺少评审凭据，或评审模型/embedding 调用失败，命令会非零退出并保存 `reports/answer-eval-failed-*.json` 部分报告，方便排查生成答案和检索上下文。

RAGAS 数据集字段为 `question`、`answer`、`contexts`、`ground_truth`。当前自动化指标包括：

- `faithfulness`：答案是否忠实于检索上下文，目标参考值 `> 0.85`。
- `answer_relevancy`：答案是否切题，目标参考值 `> 0.80`。
- `context_precision`：检索上下文是否相关，目标参考值 `> 0.75`。
- `context_recall`：是否召回支撑标准答案的关键信息，目标参考值 `> 0.80`。

本地确定性指标包括关键词覆盖率、是否给出引用、是否为空答案、是否识别未知问题、负例是否正确拒答和综合 pass rate。它们用于诊断，不替代 RAGAS 自动化评估结论。

## 如何谈评估

检索指标回答“系统有没有把正确材料找回来”，RAGAS 答案指标回答“模型是否基于材料生成了忠实、相关、上下文充分的答案”。前者更稳定、成本更低，适合反复比较 dense、hybrid、rerank、parent-child、query expansion 等检索配置；后者更接近用户体验，但会受评审 LLM、prompt、拒答策略和生成随机性影响。

可以陈述已经具备的能力：仓库有可校验的 60 条评测集、可复现实验脚本、变体对比报告、逐 case 检索证据和基于 RAGAS 的答案级自动化评估流水线。不能声称固定百分比提升、线上吞吐、P95 延迟或生产可用性，除非这些数字来自当前仓库的报告或额外压测记录。

## 模型预热

本地 embedding 和必经 reranker 首次加载 `BAAI/bge-m3`、`BAAI/bge-reranker-v2-m3` 可能较慢。演示前可以先运行下面的命令，它会构造真实 retriever 并实际触发一次检索和 rerank：

```bash
python -m scripts.warmup
```

## 端到端验证

完成 `.env` 配置并填入真实 chat 模型凭据后，按下面顺序验证完整链路：

```bash
python -m scripts.ingest
```

成功时会输出类似：

```text
IngestResult(loaded_documents=1, indexed_chunks=1, skipped=[], errors={})
```

使用 `EMBEDDING_MODEL=bge-m3` 时，这一步会在本地加载 `BAAI/bge-m3` 并生成向量；首次运行可能会从 Hugging Face 下载模型。

然后启动 API 服务：

```bash
uvicorn app.main:app --reload
```

再请求问答接口：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG 系统包含哪些核心步骤？","top_k":4}'
```

期望响应包含自然语言 `answer`，以及至少一个来自 `data/documents/example.md` 的 `sources` 引用。

本项目已验证的组合：

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-pro
EMBEDDING_MODEL=bge-m3
```

本地无模型凭据时仍可验证不依赖外部服务的部分：

```bash
pytest -v
python -c "from app.main import app; print(app.title)"
uvicorn app.main:app --host 127.0.0.1 --port 8001
curl http://127.0.0.1:8001/health
```

## 后续增强

下一阶段可以加入 Milvus/Redis 等生产化基础设施、更完整的线上监控，以及把 RAGAS 报告接入 CI 趋势看板。
