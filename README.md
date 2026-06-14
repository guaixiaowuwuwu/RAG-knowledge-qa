# RAG 知识问答系统

这是一个用于复现面试项目的 RAG 知识问答系统。它支持本地文档导入、分块、向量化、Chroma + BM25 混合检索、RRF 融合、BGE reranker 精排、LLM 回答生成和引用来源返回。

## 功能范围

- 支持 `txt`、`md`、`pdf`、`docx`、`html` 文档。
- 使用递归文本分块。
- 默认使用本地 `bge-m3` embedding 模型。
- 使用 Chroma 本地向量库和 BM25 关键词检索。
- 使用 RRF 融合 dense 与 BM25 候选结果。
- 使用 `BAAI/bge-reranker-v2-m3` 作为必经 reranker 精排模型。
- 使用 FastAPI 提供索引和问答接口。
- 返回答案和引用片段，支持普通 JSON 和 SSE 流式回答。

## 第二阶段能力

- 支持 `docx`、`html`、`htm` 文档解析。
- 建立索引时会同时生成 Chroma 向量索引和 BM25 JSONL 语料。
- 问答默认走混合检索：Chroma 稠密检索 + BM25 稀疏检索 + RRF 融合。
- 混合检索候选结果会经过 BGE Reranker 精排。
- `/ask/stream` 支持 SSE 流式输出。
- `python -m scripts.evaluate` 可运行轻量级检索评估。
- `python -m scripts.warmup` 可预热本地 embedding 和 reranker 模型。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

编辑 `.env`，填入可用的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `CHAT_MODEL`。

默认 `EMBEDDING_MODEL=bge-m3`，会在本地通过 `sentence-transformers` 加载 `BAAI/bge-m3` 生成向量，不需要配置 embedding API URL。问答检索链路还会加载 `BAAI/bge-reranker-v2-m3` 进行精排。首次运行会下载模型权重，耗时取决于网络和机器性能。

如果使用 DeepSeek 做回答生成，可以配置：

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-pro
EMBEDDING_MODEL=bge-m3
```

## 建立索引

```bash
python -m scripts.ingest
```

## 启动服务

```bash
uvicorn app.main:app --reload
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

如果 `8000` 端口已被占用，可以换一个端口启动服务：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001
curl http://127.0.0.1:8001/health
```

## 测试

```bash
pytest
```

## 轻量评估

```bash
python -m scripts.ingest
python -m scripts.evaluate
```

评估脚本会复用真实问答检索链路，包括 Chroma 稠密检索、BM25、RRF 和必经 BGE reranker。

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

下一阶段可以加入 RAGAS 评估、Web UI 和更完整的线上监控。
