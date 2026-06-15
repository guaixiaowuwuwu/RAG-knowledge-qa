# 面试讲解指南

本文档用于把项目讲成一个可验证的工程作品，而不是模板功能清单。所有指标都引用仓库内报告；面试前如果重新跑了评估，应同步更新本页的数字和报告路径。

## 1 分钟版本

我做的是一个企业知识库 RAG 问答系统，本地可以完成文档入库、混合检索、重排、父子块上下文展开、LLM 生成、流式回答和引用溯源。

离线部分支持 PDF、DOCX、HTML、Markdown、TXT，入库时会清洗文本、保留页码/标题 metadata、把 DOCX/HTML 表格转成 Markdown，并生成 child chunks、parent chunks、Chroma 向量索引和 BM25 语料。在线部分会做 Query Rewrite/HyDE，分别走 Chroma dense retrieval 和 BM25 sparse retrieval，再用 RRF 融合、BGE reranker 精排，最后组装 prompt 调 LLM，并返回答案、引用和 debug trace。

我没有复用模板里的虚高指标。当前检索对比报告是 `reports/retrieval-comparison-phase6-local.json`，数据集 `data/eval/sample_eval.jsonl` 有 60 条样本，其中 52 条正例、8 条负例，覆盖中文 BYD 公开报告和英文 SEC 10-K。Top-5 下 dense baseline 的 hit rate 是 1.000，hybrid-rerank-parent 是 0.981，MRR 从 hybrid 的 0.923 提升到 parent 版本的 0.942。这个结果也暴露了问题：负例拒答率还是 0.000，说明生产前还需要显式的相似度阈值和拒答策略。

## 3 分钟版本

这个项目的目标不是堆 RAG 名词，而是把企业知识库问答链路做成可演示、可评估、可解释的闭环。

离线入库从 `scripts.ingest` 开始。Loader 支持 PDF、DOCX、HTML、Markdown 和 TXT。PDF 会按页保留 page metadata；DOCX 和 HTML 会提取标题和表格，表格转 Markdown；文本会做基础清洗。随后系统生成两套 chunk：较小的 child chunk 用于召回，较大的 parent chunk 用于回答上下文。child chunk 写入 Chroma，原文同时写入 BM25 JSONL，parent chunk 写入 parent corpus JSONL。

在线问答从 `/ask` 或 `/ask/stream` 进入。系统先按配置做 Query Rewrite 和 HyDE，得到多个 query variants；每个 variant 同时走 dense retrieval 和 BM25；候选结果用 RRF 融合，因为 RRF 不依赖不同检索器的分数尺度，适合把向量相似度和 BM25 排名合并；融合后用 `BAAI/bge-reranker-v2-m3` 精排；命中 child chunk 后用 parent store 展开上下文；最后构造 RAG prompt 调 OpenAI-compatible chat model。前端可以看 streaming answer、citation cards、corpus status、evaluation report 和 debug trace。

我为这个系统补了评估骨架。检索评估脚本 `scripts.evaluate` 支持 dense、hybrid、hybrid-rerank、hybrid-rerank-parent、full 变体，输出 Hit Rate@K、MRR@K、Source Recall@K、Precision@K、NDCG@K 和 negative rejection。当前报告 `reports/retrieval-comparison-phase6-local.json` 基于 60 条样本：33 条中文、27 条英文；类别包含 exact fact、summary、policy、risk、comparison、SEC filing 和 negative。这个报告的结论不是“混合检索必然优于 dense”，事实是这个小数据集上 dense 很强，hybrid 引入了更多候选但排序略降；reranker 和 parent-child 主要改善 MRR/NDCG，说明它们对排序质量有帮助，但不是万能提升。

答案级评估由 `scripts.evaluate_answers` 生成，记录问题、标准答案、模型答案、上下文、引用和模型配置，并强制走 RAGAS。当前可引用的 RAGAS smoke 报告是 `reports/answer-eval-ragas-optimized.json`，只有 1 条样本，所以只能证明流水线跑通：faithfulness 1.000、answer relevancy 0.968、context precision 0.806、context recall 1.000。不能把它当成总体答案质量结论。

## 5 分钟版本

我的设计重点是三个：第一，真实企业语料和可复现评估；第二，检索链路可拆解、可观察、可调参；第三，明确区分本地 demo 和生产系统。

从数据上看，项目里包含 BYD 中文公开文档和 Apple、Microsoft、NVIDIA、Amazon、Alphabet 的 SEC 10-K HTML 文件。评测集不是随手问几个问题，而是 JSONL 标注集，字段包括 `id`、`question`、`ground_truth`、`expected_sources`、`expected_answer_keywords`，负例用 `is_negative=true` 标记。这样检索评估可以自动判断 Top-K 是否命中标注来源，也能保存逐 case retrieved sources，方便复盘失败样本。

从工程链路看，系统把 RAG 分成可替换组件：loader、cleaner、chunker、embedding、Chroma、BM25、RRF、reranker、parent store、query transformer、LLM、cache、evaluation。默认本地 embedding 是 `bge-m3`，本地向量库是 Chroma。Chroma 是为了本地复现和演示速度，不是生产规模承诺；生产迁移路径在 `docs/deployment.md` 里写成 Milvus/Qdrant adapter，而不是假装已经压测过。

检索链路的可观察性是我特别加的。`debug=true` 时 API 会返回 query variants、dense candidates、BM25 candidates、RRF scores、reranker scores、parent hydration、final chunks 和 timings。面试里如果被问“检索效果不好怎么排查”，我可以直接展示同一个问题在 dense、BM25、RRF、reranker、parent hydration 每一步发生了什么，而不是只说调 prompt。

评估结果也有不完美的地方，这反而是项目可信的地方。当前 retrieval comparison 里 dense 的 hit rate@5 是 1.000，hybrid-rerank-parent 是 0.981；如果只看命中率，混合链路没有赢。但 hybrid-rerank-parent 的 MRR 是 0.942，高于 hybrid 的 0.923，NDCG 是 0.946，高于 hybrid 的 0.932，说明 reranker/parent-child 对排序有帮助。负例拒答率全是 0.000，这是一个明确待改进点：仅靠“有没有返回检索结果”无法处理今天股价、未披露合同、实时产量这类问题，需要引入置信阈值、query intent 分类和答案层拒答检查。

生产化方面，仓库有 Dockerfile、Compose、本地 Redis 答案缓存抽象、问题长度限制、LLM timeout、Query Rewrite/HyDE timeout、请求日志、corpus status 和 latency benchmark。我的表述会很克制：这些是生产 readiness scaffold，不是线上 SLA。真正上线还需要权限过滤、异步入库、索引版本、向量库服务化、OpenTelemetry、统一异常处理、压测和质量监控。

## 架构讲解顺序

1. 先画两条链路：offline ingestion 和 online QA。对应文档是 `docs/architecture.md`。
2. 讲离线入库：文档解析、清洗、表格 Markdown、child chunk、parent chunk、embedding、Chroma、BM25 JSONL。
3. 讲在线问答：query expansion、dense/BM25、RRF、reranker、parent hydration、prompt assembly、LLM、SSE、citations。
4. 打开前端：展示 corpus status、运行评估、问一个 BYD 中文问题、问一个 SEC 英文问题、打开 debug trace。
5. 展示报告：`reports/retrieval-comparison-phase6-local.json` 和 `reports/answer-eval-ragas-optimized.json`。
6. 主动讲边界：本地 Chroma，不声明生产 QPS；RAGAS smoke 样本少，不声明总体答案质量；负例拒答还需要加强。

## 我亲自完成的内容

可以按这个口径讲，避免泛泛地说“我做了 RAG 系统”：

- 搭建 FastAPI + static frontend 的本地 demo：入库、问答、SSE streaming、引用展示、评估报告、corpus status。
- 实现和串联 RAG 核心链路：Chroma dense retrieval、BM25、RRF、BGE reranker、parent-child retrieval、Query Rewrite/HyDE。
- 扩展文档理解：支持 PDF/DOCX/HTML/Markdown/TXT，加入清洗、标题 metadata、DOCX/HTML 表格转 Markdown、PDF 表格可选抽取和表格感知分块。
- 建立可复现评估：60 条 JSONL 标注集、检索多变体对比、逐 case retrieved sources、答案级 RAGAS pipeline 和本地诊断指标。
- 补齐生产化边界文档：Docker/Compose、Redis cache 设计、请求限制、timeout、降级策略、观测建议和本地 benchmark 边界。

## 常见面试问题

### 分块大小怎么选？

分块不是越小越好。小 chunk 召回更精准，但容易缺上下文；大 chunk 上下文完整，但 embedding 噪声更高、Top-K 成本更高。这个项目用 child chunk 做召回、parent chunk 做回答上下文，就是为了兼顾精确召回和完整生成。真实项目里我会用评测集比较 chunk size、overlap、parent size，对 Hit Rate、MRR、答案 faithfulness 和延迟一起看。

### 检索效果不好怎么排查？

我会按链路拆：先看 query 是否需要改写，再看 dense 和 BM25 各自 Top-K 是否命中，之后看 RRF 有没有把正确结果融合进候选，再看 reranker 是否把正确结果排高，最后看 parent hydration 是否引入了足够上下文。这个项目的 debug trace 能直接返回这些中间结果，所以不是黑盒排查。

### 为什么需要 BM25？

向量检索擅长语义相近，BM25 擅长精确词、编号、制度名、财务科目、公司名和英文缩写。企业文档里很多问题依赖关键词精确命中，例如“募集资金三方监管协议”或 Form 10-K 的 item 名称。BM25 也不是永远提升；当前 60 条数据上 dense baseline 很强，hybrid 的 hit rate 反而略低，所以我会把 BM25 当作可评估的召回补充，而不是教条。

### Reranker 带来什么代价？

Reranker 的收益是改善排序，把真正相关的片段提前；代价是模型加载慢、推理耗时增加、候选越多成本越高，还可能因为 reranker 模型偏好导致误排。这个项目里 reranker 是默认链路，也支持 benchmark 变体对比。上线时我会限制 rerank 候选数、批处理、缓存高频问题，并监控 rerank latency。

### 如何评估 RAG？

我会分两层。检索层看 Hit Rate@K、MRR@K、Source Recall@K、Precision@K、NDCG@K 和负例拒答；答案层看 faithfulness、answer relevancy、context precision、context recall，再用本地诊断看关键词覆盖、引用率和未知问题处理。检索评估更稳定，适合频繁调参；答案评估更接近用户体验，但受评审模型和生成随机性影响。

### 如何处理表格和图片？

表格不能简单当普通段落切碎。当前项目把 DOCX/HTML 表格转成 Markdown，PDF 表格在安装 `pdfplumber` 后可抽取，并在 chunker 里尽量保持小表完整、超长表按行切分且重复表头。图片/OCR 目前明确是可选增强，不作为默认能力吹嘘；如果生产需要扫描 PDF，会引入 OCR、版面分析和人工抽检。

### 如何部署到生产？

当前仓库有 Docker/Compose、本地 Redis profile 和生产化文档，但我不会说它已经生产验证。生产版需要把 Chroma 替换或抽象到 Milvus/Qdrant/pgvector，把 BM25 换成 Elasticsearch/OpenSearch，把同步入库改成队列任务，加权限过滤、索引版本、回滚、限流、统一异常处理、OpenTelemetry 和压测。

### 如何控制成本和延迟？

先减少不必要调用：限制问题长度、Top-K、query variants 和 reranker candidate 数；Query Rewrite/HyDE 设置 timeout，失败回退原始问题；高频问题用 Redis 缓存最终答案或检索结果；本地 embedding/reranker 预热；按问题类型决定是否启用 HyDE、reranker 和 parent hydration。最后用 benchmark 和 traces 看瓶颈在 retrieval、rerank 还是 LLM。

## 最终 Demo 脚本

### 0. 准备环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

`.env` 至少配置：

```dotenv
OPENAI_API_KEY=replace-with-your-api-key
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-pro
EMBEDDING_MODEL=bge-m3
```

### 1. 入库 BYD 语料

```bash
DOCUMENTS_DIR=data/documents/byd_chinese python -m scripts.ingest
python -m scripts.warmup
```

### 2. 运行评估

```bash
python -m scripts.evaluate --validate-dataset --check-source-files
python -m scripts.evaluate --compare --top-k 5
python -m scripts.evaluate_answers --limit 10
```

如果面试现场时间有限，优先展示已保存报告：

```bash
jq '.table' reports/retrieval-comparison-phase6-local.json
jq '{summary, ragas}' reports/answer-eval-ragas-optimized.json
```

### 3. 启动服务

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/`。

### 4. 三个成功问题

```text
比亚迪 2024 年年度报告披露的营业额和同比增幅是多少？
比亚迪募集资金管理制度如何要求募集资金专户存储？
What risk does NVIDIA disclose about suppliers or manufacturing partners in its 2026 Form 10-K?
```

演示要点：答案是否引用来源、citation card 是否有 source/page/chunk/content type、debug trace 是否能看到 query variants、dense/BM25/RRF/reranker/parent hydration。

### 5. 一个负例问题

```text
比亚迪今天的股票收盘价是多少？
```

正确讲法：这是当前系统的薄弱点之一。评估报告显示 negative rejection rate 仍为 0.000，后续要加入检索置信阈值、时间敏感意图识别和答案层拒答策略。

## 不要这么说

- 不要说“召回率提升 30%+”，除非重新跑的报告确实支持。
- 不要说“生产可承载高 QPS”或“P95 达到多少”，仓库里没有生产压测。
- 不要把 1 条样本的 RAGAS smoke 分数说成整体答案质量。
- 不要说已经完整支持图片理解；当前只是文档表格增强，OCR/图片是可选方向。
