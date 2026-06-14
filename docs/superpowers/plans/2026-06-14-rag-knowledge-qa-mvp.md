# RAG Knowledge QA MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable first-stage RAG knowledge QA system with document ingestion, chunking, Chroma vector retrieval, LLM answer generation, source citations, API endpoints, and tests.

**Architecture:** The app uses FastAPI routes over focused application modules. Ingestion reads local `txt`, `md`, and `pdf` files, chunks them, embeds chunks, and persists them into Chroma. Asking a question retrieves relevant chunks, builds a grounded prompt, calls an OpenAI-compatible chat model, and returns the answer with citations.

**Tech Stack:** Python 3.11, FastAPI, Pydantic Settings, LangChain text splitter, Chroma, OpenAI-compatible API client, pypdf, pytest.

---

## File Structure

- Create: `pyproject.toml` for package metadata, dependencies, pytest config.
- Create: `.env.example` for runtime configuration.
- Create: `README.md` for setup, ingestion, API usage, and project explanation.
- Create: `app/__init__.py` package marker.
- Create: `app/main.py` FastAPI app entry point.
- Create: `app/api/__init__.py` package marker.
- Create: `app/api/schemas.py` request and response models.
- Create: `app/api/routes.py` health, ingest, and ask endpoints.
- Create: `app/core/__init__.py` package marker.
- Create: `app/core/config.py` settings loader.
- Create: `app/ingestion/__init__.py` package marker.
- Create: `app/ingestion/chunker.py` chunk model and recursive chunking.
- Create: `app/ingestion/loaders.py` document loader model and file parsers.
- Create: `app/ingestion/pipeline.py` indexing orchestration.
- Create: `app/rag/__init__.py` package marker.
- Create: `app/rag/embeddings.py` OpenAI-compatible embedding wrapper.
- Create: `app/rag/llm.py` OpenAI-compatible chat wrapper.
- Create: `app/rag/prompts.py` RAG prompt builder.
- Create: `app/rag/service.py` question answering service.
- Create: `app/rag/vector_store.py` Chroma wrapper.
- Create: `scripts/__init__.py` package marker.
- Create: `scripts/ingest.py` command-line ingestion entry point.
- Create: `data/documents/example.md` sample knowledge document.
- Create: `data/chroma/.gitkeep` placeholder for local persisted vector DB directory.
- Create: `tests/test_chunker.py` chunker tests.
- Create: `tests/test_loaders.py` loader tests.
- Create: `tests/test_rag_service.py` RAG orchestration tests.

## Task 1: Project Skeleton And Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `README.md`
- Create: `app/__init__.py`
- Create: `app/core/__init__.py`
- Create: `app/core/config.py`
- Create: `data/documents/example.md`
- Create: `data/chroma/.gitkeep`

- [ ] **Step 1: Create dependency and test configuration**

Write `pyproject.toml`:

```toml
[project]
name = "rag-knowledge-qa-system"
version = "0.1.0"
description = "A reproducible RAG knowledge QA system for interview practice."
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.5.0",
    "fastapi>=0.111.0",
    "langchain-text-splitters>=0.2.0",
    "openai>=1.30.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.2.0",
    "pypdf>=4.2.0",
    "python-dotenv>=1.0.1",
    "uvicorn[standard]>=0.29.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create environment template**

Write `.env.example`:

```dotenv
OPENAI_API_KEY=replace-with-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
DOCUMENTS_DIR=data/documents
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=rag_knowledge_base
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_TOP_K=4
```

- [ ] **Step 3: Create settings loader**

Write `app/core/config.py`:

```python
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    documents_dir: Path = Field(default=Path("data/documents"), alias="DOCUMENTS_DIR")
    chroma_dir: Path = Field(default=Path("data/chroma"), alias="CHROMA_DIR")
    chroma_collection: str = Field(default="rag_knowledge_base", alias="CHROMA_COLLECTION")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=4, alias="RETRIEVAL_TOP_K")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Create package markers**

Write empty package marker files:

```text
app/__init__.py
app/core/__init__.py
```

- [ ] **Step 5: Create sample document and vector directory placeholder**

Write `data/documents/example.md`:

```markdown
# RAG 知识问答系统

RAG 知识问答系统用于把企业内部文档转换为可检索的知识库。

系统包含文档解析、文本分块、向量化、向量检索、提示词组装和大模型回答生成六个核心步骤。

第一阶段系统支持 Markdown、TXT 和 PDF 文档。用户提问后，系统会检索最相关的文档片段，并要求大模型只基于检索到的上下文回答。

为了让答案可追溯，接口会返回引用来源，包括文件路径、页码和 chunk 编号。
```

Create `data/chroma/.gitkeep` as an empty file.

- [ ] **Step 6: Create initial README**

Write `README.md`:

```markdown
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
```

- [ ] **Step 7: Verify configuration imports**

Run:

```bash
python -c "from app.core.config import get_settings; print(get_settings().chroma_collection)"
```

Expected: prints `rag_knowledge_base`.

- [ ] **Step 8: Commit if repository is initialized**

If the directory has a `.git` folder, run:

```bash
git add pyproject.toml .env.example README.md app data
git commit -m "chore: initialize rag project skeleton"
```

If not initialized, skip the commit and record that the current directory is not a git repository.

## Task 2: Document Loading

**Files:**
- Create: `app/ingestion/__init__.py`
- Create: `app/ingestion/loaders.py`
- Create: `tests/test_loaders.py`

- [ ] **Step 1: Write failing loader tests**

Write `tests/test_loaders.py`:

```python
from pathlib import Path

import pytest

from app.ingestion.loaders import LoadedDocument, load_document, load_documents_from_dir


def test_load_markdown_document(tmp_path: Path):
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nKnowledge text.", encoding="utf-8")

    docs = load_document(path)

    assert docs == [
        LoadedDocument(
            text="# Title\n\nKnowledge text.",
            source=str(path),
            metadata={"file_type": ".md"},
        )
    ]


def test_load_text_document(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("Plain knowledge text.", encoding="utf-8")

    docs = load_document(path)

    assert docs[0].text == "Plain knowledge text."
    assert docs[0].source == str(path)
    assert docs[0].metadata == {"file_type": ".txt"}


def test_unsupported_document_type_raises(tmp_path: Path):
    path = tmp_path / "notes.csv"
    path.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(path)


def test_load_documents_from_dir_skips_unsupported_files(tmp_path: Path):
    supported = tmp_path / "supported.md"
    unsupported = tmp_path / "unsupported.csv"
    supported.write_text("Supported text", encoding="utf-8")
    unsupported.write_text("Unsupported text", encoding="utf-8")

    result = load_documents_from_dir(tmp_path)

    assert len(result.documents) == 1
    assert result.documents[0].text == "Supported text"
    assert result.skipped == [str(unsupported)]
    assert result.errors == {}
```

- [ ] **Step 2: Run loader tests and verify failure**

Run:

```bash
pytest tests/test_loaders.py -v
```

Expected: FAIL because `app.ingestion.loaders` does not exist.

- [ ] **Step 3: Implement loaders**

Write `app/ingestion/__init__.py` as an empty file.

Write `app/ingestion/loaders.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class LoadedDocument:
    text: str
    source: str
    metadata: dict


@dataclass(frozen=True)
class LoadResult:
    documents: list[LoadedDocument]
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def load_document(path: Path) -> list[LoadedDocument]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix}")

    if suffix in {".txt", ".md"}:
        return [
            LoadedDocument(
                text=path.read_text(encoding="utf-8"),
                source=str(path),
                metadata={"file_type": suffix},
            )
        ]

    return _load_pdf(path)


def load_documents_from_dir(directory: Path) -> LoadResult:
    documents: list[LoadedDocument] = []
    skipped: list[str] = []
    errors: dict[str, str] = {}

    if not directory.exists():
        return LoadResult(documents=[], skipped=[], errors={str(directory): "Directory does not exist"})

    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append(str(path))
            continue

        try:
            documents.extend(load_document(path))
        except Exception as exc:
            errors[str(path)] = str(exc)

    return LoadResult(documents=documents, skipped=skipped, errors=errors)


def _load_pdf(path: Path) -> list[LoadedDocument]:
    reader = PdfReader(str(path))
    documents: list[LoadedDocument] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        documents.append(
            LoadedDocument(
                text=text,
                source=str(path),
                metadata={"file_type": ".pdf", "page": page_index},
            )
        )

    return documents
```

- [ ] **Step 4: Run loader tests and verify pass**

Run:

```bash
pytest tests/test_loaders.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository is initialized**

```bash
git add app/ingestion tests/test_loaders.py
git commit -m "feat: add document loaders"
```

## Task 3: Text Chunking

**Files:**
- Create: `app/ingestion/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing chunker tests**

Write `tests/test_chunker.py`:

```python
from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.loaders import LoadedDocument


def test_chunk_documents_preserves_metadata():
    document = LoadedDocument(
        text="第一段内容。\n\n第二段内容。",
        source="data/documents/example.md",
        metadata={"file_type": ".md"},
    )

    chunks = chunk_documents([document], chunk_size=20, chunk_overlap=4)

    assert chunks
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].source == "data/documents/example.md"
    assert chunks[0].metadata["file_type"] == ".md"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].content.strip()


def test_chunk_documents_skips_empty_text():
    document = LoadedDocument(text="   ", source="empty.md", metadata={"file_type": ".md"})

    chunks = chunk_documents([document], chunk_size=20, chunk_overlap=4)

    assert chunks == []


def test_chunk_indexes_are_sequential_per_document():
    document = LoadedDocument(
        text="abcdefg hijklmn opqrstu vwxyz " * 5,
        source="alphabet.txt",
        metadata={"file_type": ".txt"},
    )

    chunks = chunk_documents([document], chunk_size=30, chunk_overlap=5)

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
```

- [ ] **Step 2: Run chunker tests and verify failure**

Run:

```bash
pytest tests/test_chunker.py -v
```

Expected: FAIL because `app.ingestion.chunker` does not exist.

- [ ] **Step 3: Implement chunker**

Write `app/ingestion/chunker.py`:

```python
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loaders import LoadedDocument


@dataclass(frozen=True)
class Chunk:
    content: str
    source: str
    metadata: dict


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks: list[Chunk] = []
    for document in documents:
        if not document.text.strip():
            continue

        parts = splitter.split_text(document.text)
        for index, part in enumerate(parts):
            content = part.strip()
            if not content:
                continue
            metadata = dict(document.metadata)
            metadata["source"] = document.source
            metadata["chunk_index"] = index
            chunks.append(Chunk(content=content, source=document.source, metadata=metadata))

    return chunks
```

- [ ] **Step 4: Run chunker tests and verify pass**

Run:

```bash
pytest tests/test_chunker.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository is initialized**

```bash
git add app/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: add document chunking"
```

## Task 4: Prompt And RAG Service With Test Doubles

**Files:**
- Create: `app/rag/__init__.py`
- Create: `app/rag/prompts.py`
- Create: `app/rag/service.py`
- Create: `tests/test_rag_service.py`

- [ ] **Step 1: Write failing RAG service tests**

Write `tests/test_rag_service.py`:

```python
from app.ingestion.chunker import Chunk
from app.rag.service import RagService


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_query = None
        self.last_top_k = None

    def similarity_search(self, query: str, top_k: int):
        self.last_query = query
        self.last_top_k = top_k
        return self.chunks[:top_k]


class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "这是基于知识库的回答。"


def test_answer_returns_no_context_message_when_retrieval_is_empty():
    retriever = FakeRetriever([])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    response = service.answer("系统是什么？", top_k=3)

    assert response.answer == "知识库中没有找到相关内容，无法基于现有资料回答。"
    assert response.sources == []
    assert llm.last_prompt is None


def test_answer_calls_llm_with_context_and_returns_sources():
    chunk = Chunk(
        content="RAG 系统包含文档解析和向量检索。",
        source="data/documents/example.md",
        metadata={"source": "data/documents/example.md", "chunk_index": 0, "file_type": ".md"},
    )
    retriever = FakeRetriever([chunk])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm)

    response = service.answer("RAG 系统包含什么？", top_k=5)

    assert retriever.last_query == "RAG 系统包含什么？"
    assert retriever.last_top_k == 5
    assert "RAG 系统包含文档解析和向量检索。" in llm.last_prompt
    assert "RAG 系统包含什么？" in llm.last_prompt
    assert response.answer == "这是基于知识库的回答。"
    assert response.sources[0].source == "data/documents/example.md"
    assert response.sources[0].chunk_index == 0
    assert response.sources[0].content == "RAG 系统包含文档解析和向量检索。"
```

- [ ] **Step 2: Run RAG service tests and verify failure**

Run:

```bash
pytest tests/test_rag_service.py -v
```

Expected: FAIL because `app.rag.service` does not exist.

- [ ] **Step 3: Implement prompt builder**

Write `app/rag/__init__.py` as an empty file.

Write `app/rag/prompts.py`:

```python
from app.ingestion.chunker import Chunk


def build_rag_prompt(question: str, chunks: list[Chunk]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page")
        page_text = f", page={page}" if page is not None else ""
        context_blocks.append(
            f"[{index}] source={chunk.source}{page_text}, chunk={chunk.metadata.get('chunk_index')}\n{chunk.content}"
        )

    context = "\n\n".join(context_blocks)
    return (
        "你是一个企业知识库问答助手。请只基于给定上下文回答问题。\n"
        "如果上下文不足以回答，请直接说知识库中没有找到相关内容。\n"
        "回答要简洁、准确，并尽量指出依据来自哪些引用编号。\n\n"
        f"上下文：\n{context}\n\n"
        f"问题：{question}\n\n"
        "答案："
    )
```

- [ ] **Step 4: Implement RAG service**

Write `app/rag/service.py`:

```python
from dataclasses import dataclass
from typing import Protocol

from app.ingestion.chunker import Chunk
from app.rag.prompts import build_rag_prompt


@dataclass(frozen=True)
class Source:
    source: str
    page: int | None
    chunk_index: int | None
    content: str


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source]


class Retriever(Protocol):
    def similarity_search(self, query: str, top_k: int) -> list[Chunk]:
        ...


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class RagService:
    def __init__(self, retriever: Retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, top_k: int) -> Answer:
        chunks = self.retriever.similarity_search(question, top_k=top_k)
        if not chunks:
            return Answer(answer="知识库中没有找到相关内容，无法基于现有资料回答。", sources=[])

        prompt = build_rag_prompt(question, chunks)
        answer = self.llm.complete(prompt)
        sources = [
            Source(
                source=chunk.source,
                page=chunk.metadata.get("page"),
                chunk_index=chunk.metadata.get("chunk_index"),
                content=chunk.content,
            )
            for chunk in chunks
        ]
        return Answer(answer=answer, sources=sources)
```

- [ ] **Step 5: Run RAG service tests and verify pass**

Run:

```bash
pytest tests/test_rag_service.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit if repository is initialized**

```bash
git add app/rag tests/test_rag_service.py
git commit -m "feat: add rag service"
```

## Task 5: Embeddings, LLM, Vector Store, And Ingestion Pipeline

**Files:**
- Create: `app/rag/embeddings.py`
- Create: `app/rag/llm.py`
- Create: `app/rag/vector_store.py`
- Create: `app/ingestion/pipeline.py`
- Create: `scripts/__init__.py`
- Create: `scripts/ingest.py`

- [ ] **Step 1: Implement embedding client**

Write `app/rag/embeddings.py`:

```python
from openai import OpenAI


class OpenAIEmbeddings:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
```

- [ ] **Step 2: Implement LLM client**

Write `app/rag/llm.py`:

```python
from openai import OpenAI


class OpenAIChatLLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for chat completion")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content or ""
```

- [ ] **Step 3: Implement Chroma vector store**

Write `app/rag/vector_store.py`:

```python
from pathlib import Path
from uuid import uuid4

import chromadb

from app.ingestion.chunker import Chunk


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embeddings):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)
        ids = [str(uuid4()) for _ in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
        return len(chunks)

    def similarity_search(self, query: str, top_k: int) -> list[Chunk]:
        vector = self.embeddings.embed_query(query)
        result = self.collection.query(query_embeddings=[vector], n_results=top_k)

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        chunks: list[Chunk] = []

        for content, metadata in zip(documents, metadatas, strict=False):
            metadata = metadata or {}
            source = str(metadata.get("source", ""))
            chunks.append(Chunk(content=content, source=source, metadata=dict(metadata)))

        return chunks
```

- [ ] **Step 4: Implement ingestion pipeline**

Write `app/ingestion/pipeline.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import load_documents_from_dir


@dataclass(frozen=True)
class IngestResult:
    loaded_documents: int
    indexed_chunks: int
    skipped: list[str]
    errors: dict[str, str]


def ingest_directory(
    documents_dir: Path,
    vector_store,
    chunk_size: int,
    chunk_overlap: int,
    reset: bool = True,
) -> IngestResult:
    load_result = load_documents_from_dir(documents_dir)
    chunks = chunk_documents(load_result.documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if reset:
        vector_store.reset()

    indexed = vector_store.add_chunks(chunks)
    return IngestResult(
        loaded_documents=len(load_result.documents),
        indexed_chunks=indexed,
        skipped=load_result.skipped,
        errors=load_result.errors,
    )
```

- [ ] **Step 5: Implement command-line ingestion script**

Write `scripts/__init__.py` as an empty file.

Write `scripts/ingest.py`:

```python
from app.core.config import get_settings
from app.ingestion.pipeline import ingest_directory
from app.rag.embeddings import OpenAIEmbeddings
from app.rag.vector_store import ChromaVectorStore


def main() -> None:
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=embeddings,
    )
    result = ingest_directory(
        documents_dir=settings.documents_dir,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        reset=True,
    )
    print(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run existing tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 7: Manually verify ingestion with real credentials**

After filling `.env`, run:

```bash
python -m scripts.ingest
```

Expected: output similar to:

```text
IngestResult(loaded_documents=1, indexed_chunks=1, skipped=[], errors={})
```

The exact `indexed_chunks` value may differ if chunk size creates multiple chunks.

- [ ] **Step 8: Commit if repository is initialized**

```bash
git add app/ingestion/pipeline.py app/rag/embeddings.py app/rag/llm.py app/rag/vector_store.py scripts
git commit -m "feat: add vector ingestion pipeline"
```

## Task 6: FastAPI Routes

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/schemas.py`
- Create: `app/api/routes.py`
- Create: `app/main.py`

- [ ] **Step 1: Implement API schemas**

Write `app/api/__init__.py` as an empty file.

Write `app/api/schemas.py`:

```python
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    source: str
    page: int | None = None
    chunk_index: int | None = None
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class IngestResponse(BaseModel):
    loaded_documents: int
    indexed_chunks: int
    skipped: list[str]
    errors: dict[str, str]
```

- [ ] **Step 2: Implement dependency factory and routes**

Write `app/api/routes.py`:

```python
from fastapi import APIRouter

from app.api.schemas import AskRequest, AskResponse, IngestResponse, SourceResponse
from app.core.config import get_settings
from app.ingestion.pipeline import ingest_directory
from app.rag.embeddings import OpenAIEmbeddings
from app.rag.llm import OpenAIChatLLM
from app.rag.service import RagService
from app.rag.vector_store import ChromaVectorStore


router = APIRouter()


def build_embeddings():
    settings = get_settings()
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )


def build_vector_store():
    settings = get_settings()
    return ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=build_embeddings(),
    )


def build_rag_service():
    settings = get_settings()
    llm = OpenAIChatLLM(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
    )
    return RagService(retriever=build_vector_store(), llm=llm)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
def ingest():
    settings = get_settings()
    result = ingest_directory(
        documents_dir=settings.documents_dir,
        vector_store=build_vector_store(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        reset=True,
    )
    return IngestResponse(
        loaded_documents=result.loaded_documents,
        indexed_chunks=result.indexed_chunks,
        skipped=result.skipped,
        errors=result.errors,
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    settings = get_settings()
    service = build_rag_service()
    answer = service.answer(
        question=request.question,
        top_k=request.top_k or settings.retrieval_top_k,
    )
    return AskResponse(
        answer=answer.answer,
        sources=[
            SourceResponse(
                source=source.source,
                page=source.page,
                chunk_index=source.chunk_index,
                content=source.content,
            )
            for source in answer.sources
        ],
    )
```

- [ ] **Step 3: Implement FastAPI app entry point**

Write `app/main.py`:

```python
from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(title="RAG Knowledge QA System")
app.include_router(router)
```

- [ ] **Step 4: Verify app imports**

Run:

```bash
python -c "from app.main import app; print(app.title)"
```

Expected: prints `RAG Knowledge QA System`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 6: Manually verify HTTP health endpoint**

Run:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

Stop the server after verification.

- [ ] **Step 7: Commit if repository is initialized**

```bash
git add app/api app/main.py
git commit -m "feat: expose rag api"
```

## Task 7: End-To-End Manual Verification And Documentation Cleanup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run all automated tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 2: Install and configure the app**

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` with real credentials.

- [ ] **Step 3: Build the vector index**

Run:

```bash
python -m scripts.ingest
```

Expected: `indexed_chunks` is greater than `0`.

- [ ] **Step 4: Start API server**

Run:

```bash
uvicorn app.main:app --reload
```

Expected: Uvicorn starts on `http://127.0.0.1:8000`.

- [ ] **Step 5: Ask a question**

Run:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG 系统包含哪些核心步骤？","top_k":4}'
```

Expected:

```json
{
  "answer": "...",
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

- [ ] **Step 6: Update README with any provider-specific note discovered during verification**

If the configured provider does not support `text-embedding-3-small`, update README to state the actual working `EMBEDDING_MODEL` used in `.env`.

- [ ] **Step 7: Commit if repository is initialized**

```bash
git add README.md
git commit -m "docs: document rag mvp verification"
```

## Plan Self-Review

- Spec coverage: The plan covers project skeleton, configuration, document loading, chunking, vector storage, ingestion, RAG answering, API routes, tests, sample data, and manual verification.
- Placeholder scan: No `TBD`, `TODO`, or undefined implementation placeholders remain.
- Type consistency: `LoadedDocument`, `Chunk`, `Source`, `Answer`, `IngestResult`, and API response schemas are consistently named across tasks.
- Scope control: BM25, RRF, Reranker, RAGAS, Web UI, and Milvus remain explicitly out of first-stage implementation.
