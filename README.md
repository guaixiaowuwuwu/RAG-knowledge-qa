# RAG 知识问答系统

这是一个用于复现面试项目的第一阶段 RAG 知识问答系统。它支持本地文档导入、分块、向量化、Chroma 检索、LLM 回答生成和引用来源返回。

## 功能范围

- 支持 `txt`、`md`、`pdf` 文档。
- 使用递归文本分块。
- 使用 OpenAI-compatible Embedding API。
- 使用 Chroma 本地向量库。
- 使用 FastAPI 提供索引和问答接口。
- 返回答案和引用片段。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

编辑 `.env`，填入可用的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`CHAT_MODEL` 和 `EMBEDDING_MODEL`。

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

## 测试

```bash
pytest
```

## 后续增强

下一阶段可以加入 BM25、RRF 融合、Reranker、RAGAS 评估、SSE 流式输出和 Web UI。
