# RAG Knowledge QA System Design

## Goal

复现一个可运行的 RAG 知识问答系统，用于面试项目展示和本地学习。第一阶段聚焦最小可用闭环：导入文档、解析分块、向量化入库、检索、调用大模型生成答案，并返回引用来源。

该项目参考 `https://zchary1106.github.io/agent-interview-hub/` 中“项目实战”的第一个项目“RAG 知识问答系统”。网页中的生产级指标和高级能力作为后续增强目标，不在第一阶段承诺实现。

## Scope

### In Scope

- Python 后端工程骨架。
- FastAPI HTTP API。
- 本地文档导入，支持 `txt`、`md`、`pdf`。
- 文档解析与基础元数据保留。
- 递归文本分块。
- Embedding 生成。
- Chroma 本地向量库。
- 基于向量相似度的 Top-K 检索。
- RAG Prompt 组装。
- LLM 问答接口。
- 回答中返回引用片段和来源。
- 命令行或 API 方式重建索引。
- 基础单元测试，覆盖分块、检索接口边界、Prompt 输入结构。
- `.env.example`、`README.md`、示例数据目录和运行说明。

### Out of Scope For First Stage

- Milvus、Elasticsearch、Redis 等外部基础设施。
- BM25 + RRF 混合检索。
- Reranker 精排。
- RAGAS 自动评估。
- 多模态图表解析。
- Web 前端。
- 用户体系、权限、租户隔离。
- 生产级监控、日志平台和部署流水线。

这些能力会在第一阶段架构中预留扩展接口，但不直接实现。

## Architecture

系统采用清晰的模块边界，而不是把所有逻辑写进 FastAPI 路由。

```text
User/API
  -> FastAPI routes
  -> Application services
  -> Document loaders / chunker / embeddings / vector store / LLM client
  -> Chroma persisted database
```

核心流程分为两条：

1. 索引流程：读取文档，解析成文本，切分为 chunk，生成 embedding，写入 Chroma。
2. 问答流程：接收问题，向量检索相关 chunk，把上下文和问题组装成 Prompt，调用 LLM，返回答案和引用。

## Proposed File Structure

```text
.
├── app
│   ├── __init__.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core
│   │   ├── __init__.py
│   │   └── config.py
│   ├── ingestion
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── loaders.py
│   │   └── pipeline.py
│   ├── rag
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   ├── service.py
│   │   └── vector_store.py
│   └── main.py
├── data
│   ├── documents
│   └── chroma
├── scripts
│   └── ingest.py
├── tests
│   ├── test_chunker.py
│   ├── test_loaders.py
│   └── test_rag_service.py
├── .env.example
├── pyproject.toml
└── README.md
```

## Components

### Configuration

`app/core/config.py` 负责读取环境变量，集中管理模型、API Key、路径、检索参数。

关键配置：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `CHAT_MODEL`
- `EMBEDDING_MODEL`
- `DOCUMENTS_DIR`
- `CHROMA_DIR`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_TOP_K`

### Document Loaders

`app/ingestion/loaders.py` 根据文件后缀选择解析方式。

第一阶段支持：

- `.txt`：直接读取 UTF-8 文本。
- `.md`：按文本读取，保留 Markdown 内容。
- `.pdf`：使用 `pypdf` 提取页文本，并把页码写入元数据。

每个解析结果统一成 `LoadedDocument`：

```python
LoadedDocument(
    text="...",
    source="data/documents/example.pdf",
    metadata={"page": 1}
)
```

### Chunker

`app/ingestion/chunker.py` 使用 LangChain 的递归字符分割器，输出带来源元数据的 chunk。

每个 chunk 至少包含：

- `content`
- `source`
- `chunk_index`
- 原始 loader metadata，例如 `page`

默认参数：

- `chunk_size=800`
- `chunk_overlap=120`

### Embeddings

`app/rag/embeddings.py` 先使用 OpenAI 兼容接口，方便接入 OpenAI、DeepSeek 兼容网关或其他 OpenAI-compatible 服务。

默认 embedding 模型：

- `text-embedding-3-small`

如果用户使用的服务不支持该模型，需要通过 `.env` 调整。

### Vector Store

`app/rag/vector_store.py` 封装 Chroma。

接口保持窄边界：

- `add_chunks(chunks)`
- `similarity_search(query, top_k)`
- `reset()`

这样后续替换 Milvus 或增加混合检索时，不需要改 API 路由。

### RAG Service

`app/rag/service.py` 负责问答主流程：

1. 接收用户问题。
2. 从向量库检索 Top-K chunk。
3. 把 chunk 拼成上下文。
4. 使用固定 Prompt 约束模型基于上下文回答。
5. 返回答案和引用。

如果检索结果为空，服务返回明确的“知识库中没有找到相关内容”，避免模型无依据编造。

### API

`app/api/routes.py` 提供第一阶段接口：

- `GET /health`：健康检查。
- `POST /ingest`：扫描文档目录并重建索引。
- `POST /ask`：输入问题，返回答案和引用。

请求示例：

```json
{
  "question": "这个系统的核心模块有哪些？",
  "top_k": 4
}
```

响应示例：

```json
{
  "answer": "系统包含文档解析、分块、向量检索和答案生成模块。",
  "sources": [
    {
      "source": "data/documents/example.md",
      "page": null,
      "chunk_index": 0,
      "content": "..."
    }
  ]
}
```

## Error Handling

- 不支持的文件类型会被跳过，并在索引结果中返回 skipped 列表。
- 文档目录为空时，`POST /ingest` 返回成功状态和 `indexed_chunks=0`。
- 缺少 API Key 时，启动或首次调用模型时返回清晰错误。
- 问答时检索不到上下文，返回无答案状态，不调用 LLM。
- 单个文档解析失败不阻断整个批量索引。

## Testing Strategy

第一阶段测试重点放在本地可验证逻辑，避免测试强依赖真实 LLM。

- `test_chunker.py`：验证 chunk overlap、metadata 传递、空文本处理。
- `test_loaders.py`：验证 txt/md 加载和不支持文件类型处理。
- `test_rag_service.py`：使用 fake retriever 和 fake LLM 验证无检索结果、有检索结果、引用返回格式。

真实 API Key 和真实 Chroma 端到端测试放在手动验证说明中。

## Manual Verification

完成第一阶段后，应能执行：

```bash
cp .env.example .env
```

填入 API Key 后：

```bash
python -m scripts.ingest
uvicorn app.main:app --reload
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"示例文档讲了什么？","top_k":4}'
```

验收标准：

- 服务能启动。
- 示例文档能入库。
- 问答接口能返回自然语言答案。
- 响应包含至少一个引用来源。
- 测试命令通过。

## Future Enhancements

第二阶段建议按以下顺序增强：

1. 增加 BM25 检索。
2. 使用 RRF 融合 BM25 和向量检索结果。
3. 增加 BGE Reranker 精排。
4. 增加 query rewrite 和 multi-query retrieval。
5. 引入 RAGAS 评估集和指标脚本。
6. 增加 SSE 流式输出。
7. 增加简洁 Web UI。
8. 把 Chroma 替换为 Milvus，并通过 Docker Compose 管理依赖。

## Open Decisions

- 第一阶段默认使用 OpenAI-compatible 接口；具体供应商由 `.env` 决定。
- 第一阶段不做前端，优先保证后端 RAG 主链路可运行。
- 第一阶段不承诺网页中的生产指标，只保留可解释、可演示、可扩展的工程结构。
