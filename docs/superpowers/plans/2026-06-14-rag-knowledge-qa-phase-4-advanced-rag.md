# RAG Knowledge QA Phase 4 Advanced RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the next major gap with the reference RAG project by adding parent-child retrieval, query rewriting/HyDE query expansion, and a more credible offline evaluation loop that can be shown in the UI.

**Architecture:** Keep the existing FastAPI + static frontend + Chroma/BM25/RRF/BGE-reranker pipeline, then add parent document storage and query transformation as optional-but-default retrieval improvements behind the existing `similarity_search(query, top_k)` contract. Evaluation will reuse the real retriever path and produce local deterministic retrieval metrics plus optional LLM-answer records for later RAGAS integration.

**Tech Stack:** Python 3.11, FastAPI, Chroma, rank-bm25, sentence-transformers, FlagEmbedding, OpenAI-compatible chat client, JSONL parent stores/evaluation sets, pytest, vanilla frontend JavaScript tests.

---

## Why This Phase

The reference project highlights several capabilities that are not fully implemented yet:

- 多级分块 + 父子文档索引。
- Query 预处理、Query 改写、多查询扩展、HyDE。
- RAGAS 自动化评估流水线和指标口径。
- 可解释的评估结果，用来支撑“不是凭感觉调参”。

The current project already has document parsing, recursive chunking, Chroma, BM25, RRF, mandatory BGE reranking, SSE streaming, citations, a frontend, and a lightweight retrieval evaluation script. Phase 4 should deepen retrieval quality and evaluation credibility before moving to Milvus/Redis/K8s production infrastructure.

## Scope

### In Scope

- Persist parent chunks alongside child chunks during ingestion.
- Store `parent_id` on child metadata.
- Return parent content to the LLM after child-level retrieval hits.
- Add query rewrite and HyDE query expansion through the existing chat LLM interface.
- Allow retrieval to run multiple query variants and merge candidates before reranking.
- Add local retrieval metrics: Hit Rate@K, MRR@K, and source recall.
- Extend evaluation JSONL schema with `expected_answer_keywords`.
- Add an evaluation report endpoint for the frontend.
- Show evaluation output in the frontend.
- Add tests for parent store, parent-child retrieval, query transforms, multi-query retrieval, metrics, endpoint, and UI utilities.

### Out of Scope

- Full RAGAS dependency and LLM-as-judge scoring.
- Milvus migration.
- Redis cache.
- K8s/Nginx deployment.
- Image OCR/VLM processing.
- Browser file upload.

RAGAS and production deployment are valuable, but they are heavier dependencies and require a better labeled dataset. This phase builds the local measurement spine first. Skeleton before armor; less dramatic, more useful.

## File Structure

- Modify: `app/core/config.py`.
- Modify: `.env.example`.
- Modify: `README.md`.
- Modify: `app/ingestion/chunker.py`.
- Modify: `app/ingestion/pipeline.py`.
- Modify: `app/rag/documents.py`.
- Create: `app/rag/parent_store.py`.
- Modify: `app/rag/hybrid_retriever.py`.
- Create: `app/rag/query_transform.py`.
- Modify: `app/api/routes.py`.
- Modify: `app/api/schemas.py`.
- Modify: `app/evaluation/dataset.py`.
- Modify: `app/evaluation/metrics.py`.
- Create: `app/evaluation/report.py`.
- Modify: `scripts/evaluate.py`.
- Modify: `app/web/static/index.html`.
- Modify: `app/web/static/app.js`.
- Modify: `app/web/static/styles.css`.
- Modify: `app/web/static/ui-utils.js`.
- Create: `tests/test_parent_store.py`.
- Create: `tests/test_parent_child_retrieval.py`.
- Create: `tests/test_query_transform.py`.
- Create: `tests/test_advanced_evaluation.py`.
- Create: `tests/test_evaluation_endpoint.py`.
- Modify: `tests/js/ui-utils.test.mjs`.

## Task 1: Advanced RAG Settings

**Files:**
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Test: `tests/test_phase4_config.py`

- [ ] **Step 1: Write failing settings test**

Create `tests/test_phase4_config.py`:

```python
from app.core.config import Settings


def test_phase4_settings_defaults():
    settings = Settings()

    assert settings.parent_corpus_path.as_posix() == "data/chroma/parent_corpus.jsonl"
    assert settings.parent_chunk_size == 2048
    assert settings.parent_chunk_overlap == 160
    assert settings.query_rewrite_enabled is True
    assert settings.hyde_enabled is True
    assert settings.max_query_variants == 4
    assert settings.eval_dataset_path.as_posix() == "data/eval/sample_eval.jsonl"
```

- [ ] **Step 2: Run settings test and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_phase4_config.py -v
```

Expected: FAIL because Phase 4 settings do not exist.

- [ ] **Step 3: Add settings fields**

Modify `app/core/config.py` by adding these fields to `Settings`:

```python
    parent_corpus_path: Path = Field(default=Path("data/chroma/parent_corpus.jsonl"), alias="PARENT_CORPUS_PATH")
    parent_chunk_size: int = Field(default=2048, alias="PARENT_CHUNK_SIZE")
    parent_chunk_overlap: int = Field(default=160, alias="PARENT_CHUNK_OVERLAP")
    query_rewrite_enabled: bool = Field(default=True, alias="QUERY_REWRITE_ENABLED")
    hyde_enabled: bool = Field(default=True, alias="HYDE_ENABLED")
    max_query_variants: int = Field(default=4, alias="MAX_QUERY_VARIANTS")
    eval_dataset_path: Path = Field(default=Path("data/eval/sample_eval.jsonl"), alias="EVAL_DATASET_PATH")
```

- [ ] **Step 4: Add environment template values**

Append to `.env.example`:

```dotenv
PARENT_CORPUS_PATH=data/chroma/parent_corpus.jsonl
PARENT_CHUNK_SIZE=2048
PARENT_CHUNK_OVERLAP=160
QUERY_REWRITE_ENABLED=true
HYDE_ENABLED=true
MAX_QUERY_VARIANTS=4
EVAL_DATASET_PATH=data/eval/sample_eval.jsonl
```

- [ ] **Step 5: Run settings test and verify pass**

Run:

```bash
.venv/bin/pytest tests/test_phase4_config.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add app/core/config.py .env.example tests/test_phase4_config.py
git commit -m "chore: add advanced rag settings"
```

## Task 2: Parent Chunking And Parent Store

**Files:**
- Modify: `app/ingestion/chunker.py`
- Create: `app/rag/parent_store.py`
- Modify: `app/rag/documents.py`
- Test: `tests/test_parent_store.py`

- [ ] **Step 1: Write failing parent store tests**

Create `tests/test_parent_store.py`:

```python
from pathlib import Path

from app.ingestion.chunker import Chunk, ParentChildChunks, chunk_documents_with_parents
from app.ingestion.loaders import LoadedDocument
from app.rag.parent_store import JsonlParentStore


def test_chunk_documents_with_parents_links_children_to_parent():
    document = LoadedDocument(
        text="第一段介绍 RAG 系统。\n\n第二段介绍混合检索。\n\n第三段介绍 reranker。",
        source="example.md",
        metadata={"file_type": ".md"},
    )

    result = chunk_documents_with_parents(
        [document],
        child_chunk_size=24,
        child_chunk_overlap=4,
        parent_chunk_size=80,
        parent_chunk_overlap=8,
    )

    assert isinstance(result, ParentChildChunks)
    assert result.parents
    assert result.children
    assert all("parent_id" in child.metadata for child in result.children)
    assert {child.metadata["parent_id"] for child in result.children}.issubset(
        {parent.metadata["parent_id"] for parent in result.parents}
    )


def test_jsonl_parent_store_round_trips_parent_chunks(tmp_path: Path):
    path = tmp_path / "parents.jsonl"
    parent = Chunk(
        content="父块上下文内容",
        source="example.md",
        metadata={"source": "example.md", "parent_id": "parent-1", "chunk_index": 0},
    )

    store = JsonlParentStore(path)
    store.write([parent])
    loaded = JsonlParentStore(path)

    result = loaded.get("parent-1")

    assert result is not None
    assert result.content == "父块上下文内容"
    assert result.metadata["parent_id"] == "parent-1"
```

- [ ] **Step 2: Run parent tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_parent_store.py -v
```

Expected: FAIL because parent chunking and parent store do not exist.

- [ ] **Step 3: Add parent-child chunk dataclass and function**

Modify `app/ingestion/chunker.py`:

```python
from dataclasses import dataclass
from hashlib import sha1
```

Keep existing `Chunk`. Add:

```python
@dataclass(frozen=True)
class ParentChildChunks:
    parents: list[Chunk]
    children: list[Chunk]


def parent_id_for(source: str, parent_index: int, content: str) -> str:
    raw = f"{source}:parent:{parent_index}:{content}"
    return sha1(raw.encode("utf-8")).hexdigest()
```

Add:

```python
def chunk_documents_with_parents(
    documents: list[LoadedDocument],
    child_chunk_size: int,
    child_chunk_overlap: int,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
) -> ParentChildChunks:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for document in documents:
        if not document.text.strip():
            continue

        parent_parts = parent_splitter.split_text(document.text)
        for parent_index, parent_part in enumerate(parent_parts):
            parent_content = parent_part.strip()
            if not parent_content:
                continue

            identity = parent_id_for(document.source, parent_index, parent_content)
            parent_metadata = dict(document.metadata)
            parent_metadata["source"] = document.source
            parent_metadata["chunk_index"] = parent_index
            parent_metadata["parent_id"] = identity
            parents.append(Chunk(content=parent_content, source=document.source, metadata=parent_metadata))

            child_parts = child_splitter.split_text(parent_content)
            for child_index, child_part in enumerate(child_parts):
                child_content = child_part.strip()
                if not child_content:
                    continue
                child_metadata = dict(document.metadata)
                child_metadata["source"] = document.source
                child_metadata["chunk_index"] = child_index
                child_metadata["parent_id"] = identity
                children.append(Chunk(content=child_content, source=document.source, metadata=child_metadata))

    return ParentChildChunks(parents=parents, children=children)
```

- [ ] **Step 4: Implement JSONL parent store**

Create `app/rag/parent_store.py`:

```python
import json
from pathlib import Path

from app.ingestion.chunker import Chunk


class JsonlParentStore:
    def __init__(self, path: Path):
        self.path = path
        self._parents = self._load()

    def _load(self) -> dict[str, Chunk]:
        if not self.path.exists():
            return {}

        parents: dict[str, Chunk] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = dict(row["metadata"])
            parent_id = str(metadata["parent_id"])
            parents[parent_id] = Chunk(
                content=str(row["content"]),
                source=str(row["source"]),
                metadata=metadata,
            )
        return parents

    def write(self, parents: list[Chunk]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for parent in parents:
                row = {
                    "content": parent.content,
                    "source": parent.source,
                    "metadata": parent.metadata,
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._parents = self._load()

    def get(self, parent_id: str) -> Chunk | None:
        return self._parents.get(parent_id)

    def hydrate(self, children: list[Chunk]) -> list[Chunk]:
        hydrated: list[Chunk] = []
        seen_parent_ids: set[str] = set()
        for child in children:
            parent_id = child.metadata.get("parent_id")
            parent = self.get(str(parent_id)) if parent_id else None
            if parent is None:
                hydrated.append(child)
                continue
            if str(parent_id) in seen_parent_ids:
                continue
            seen_parent_ids.add(str(parent_id))
            metadata = dict(parent.metadata)
            metadata["matched_child_chunk_index"] = child.metadata.get("chunk_index")
            hydrated.append(Chunk(content=parent.content, source=parent.source, metadata=metadata))
        return hydrated
```

- [ ] **Step 5: Run parent tests and verify pass**

Run:

```bash
.venv/bin/pytest tests/test_parent_store.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add app/ingestion/chunker.py app/rag/parent_store.py tests/test_parent_store.py
git commit -m "feat: add parent child chunk storage"
```

## Task 3: Persist Parent Corpus During Ingestion

**Files:**
- Modify: `app/ingestion/pipeline.py`
- Modify: `app/api/routes.py`
- Modify: `scripts/ingest.py`
- Test: `tests/test_pipeline_parent_corpus.py`

- [ ] **Step 1: Write failing pipeline test**

Create `tests/test_pipeline_parent_corpus.py`:

```python
from pathlib import Path

from app.ingestion.pipeline import ingest_directory


class FakeVectorStore:
    def __init__(self):
        self.reset_called = False
        self.chunks = []

    def reset(self):
        self.reset_called = True

    def add_chunks(self, chunks):
        self.chunks = chunks
        return len(chunks)


def test_ingest_directory_persists_parent_corpus(tmp_path: Path):
    documents = tmp_path / "docs"
    documents.mkdir()
    (documents / "guide.md").write_text(
        "第一段介绍 RAG。\n\n第二段介绍 BM25。\n\n第三段介绍 Reranker。",
        encoding="utf-8",
    )
    parent_path = tmp_path / "parents.jsonl"
    bm25_path = tmp_path / "bm25.jsonl"

    vector_store = FakeVectorStore()
    result = ingest_directory(
        documents_dir=documents,
        vector_store=vector_store,
        chunk_size=24,
        chunk_overlap=4,
        reset=True,
        bm25_corpus_path=bm25_path,
        parent_corpus_path=parent_path,
        parent_chunk_size=80,
        parent_chunk_overlap=8,
    )

    assert result.indexed_chunks > 0
    assert parent_path.exists()
    assert bm25_path.exists()
    assert all("parent_id" in chunk.metadata for chunk in vector_store.chunks)
```

- [ ] **Step 2: Run pipeline test and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_pipeline_parent_corpus.py -v
```

Expected: FAIL because `ingest_directory` does not accept parent corpus arguments.

- [ ] **Step 3: Update ingestion pipeline**

Modify imports in `app/ingestion/pipeline.py`:

```python
from app.ingestion.chunker import Chunk, chunk_documents, chunk_documents_with_parents
from app.rag.parent_store import JsonlParentStore
```

Change `ingest_directory` signature:

```python
    parent_corpus_path: Path | None = None,
    parent_chunk_size: int | None = None,
    parent_chunk_overlap: int | None = None,
```

Replace chunking line with:

```python
    if parent_corpus_path is not None and parent_chunk_size is not None and parent_chunk_overlap is not None:
        parent_child = chunk_documents_with_parents(
            load_result.documents,
            child_chunk_size=chunk_size,
            child_chunk_overlap=chunk_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
        )
        chunks = parent_child.children
        JsonlParentStore(parent_corpus_path).write(parent_child.parents)
    else:
        chunks = chunk_documents(load_result.documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
```

- [ ] **Step 4: Pass parent settings from API**

Modify `app/api/routes.py` ingestion call:

```python
        parent_corpus_path=settings.parent_corpus_path,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
```

- [ ] **Step 5: Pass parent settings from script**

Modify `scripts/ingest.py` ingestion call:

```python
        parent_corpus_path=settings.parent_corpus_path,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
```

- [ ] **Step 6: Run pipeline tests**

Run:

```bash
.venv/bin/pytest tests/test_pipeline_parent_corpus.py tests/test_pipeline_corpus.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add app/ingestion/pipeline.py app/api/routes.py scripts/ingest.py tests/test_pipeline_parent_corpus.py
git commit -m "feat: persist parent corpus during ingestion"
```

## Task 4: Parent-Aware Hybrid Retrieval

**Files:**
- Modify: `app/rag/hybrid_retriever.py`
- Modify: `app/api/routes.py`
- Test: `tests/test_parent_child_retrieval.py`

- [ ] **Step 1: Write failing parent retrieval test**

Create `tests/test_parent_child_retrieval.py`:

```python
from app.ingestion.chunker import Chunk
from app.rag.documents import RetrievedDocument
from app.rag.hybrid_retriever import HybridRetriever


class FakeDenseRetriever:
    def similarity_search(self, query: str, top_k: int):
        return [
            Chunk(
                content="子块：BM25",
                source="guide.md",
                metadata={"source": "guide.md", "chunk_index": 0, "parent_id": "parent-1"},
            )
        ]


class FakeSparseRetriever:
    def search(self, query: str, top_k: int):
        return []


class FakeParentStore:
    def hydrate(self, chunks):
        return [
            Chunk(
                content="父块：RAG 系统包含 BM25、RRF 和 Reranker 的完整上下文。",
                source="guide.md",
                metadata={"source": "guide.md", "chunk_index": 0, "parent_id": "parent-1"},
            )
        ]


class FakeReranker:
    def rerank(self, query, documents, top_n):
        return documents[:top_n]


def test_hybrid_retriever_hydrates_parent_context_before_rerank():
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        sparse_retriever=FakeSparseRetriever(),
        reranker=FakeReranker(),
        dense_top_k=1,
        sparse_top_k=1,
        rrf_k=60,
        reranker_top_n=1,
        parent_store=FakeParentStore(),
    )

    chunks = retriever.similarity_search("BM25 是什么？", top_k=1)

    assert chunks[0].content.startswith("父块")
```

- [ ] **Step 2: Run parent retrieval test and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_parent_child_retrieval.py -v
```

Expected: FAIL because `HybridRetriever` does not accept `parent_store`.

- [ ] **Step 3: Add parent store support to hybrid retriever**

Modify `app/rag/hybrid_retriever.py` constructor:

```python
        parent_store=None,
```

Set:

```python
        self.parent_store = parent_store
```

In `similarity_search`, after fusion and before reranking, add:

```python
        fused_chunks = [retrieved_document_to_chunk(document) for document in fused]
        if self.parent_store is not None:
            fused_chunks = self.parent_store.hydrate(fused_chunks)
        fused = [chunk_to_retrieved_document(chunk) for chunk in fused_chunks]
```

Keep reranking after hydration:

```python
        reranked = self.reranker.rerank(query, fused, top_n=max(top_k, self.reranker_top_n))
```

- [ ] **Step 4: Wire parent store in API factory**

Modify `app/api/routes.py` import:

```python
from app.rag.parent_store import JsonlParentStore
```

In `build_retriever`, create:

```python
    parent_store = JsonlParentStore(settings.parent_corpus_path)
```

Pass into `HybridRetriever`:

```python
        parent_store=parent_store,
```

- [ ] **Step 5: Run retrieval tests**

Run:

```bash
.venv/bin/pytest tests/test_parent_child_retrieval.py tests/test_hybrid_retriever.py tests/test_api_retriever_factory.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add app/rag/hybrid_retriever.py app/api/routes.py tests/test_parent_child_retrieval.py
git commit -m "feat: hydrate parent context in retrieval"
```

## Task 5: Query Rewrite And HyDE

**Files:**
- Create: `app/rag/query_transform.py`
- Modify: `app/rag/hybrid_retriever.py`
- Modify: `app/api/routes.py`
- Test: `tests/test_query_transform.py`

- [ ] **Step 1: Write failing query transform tests**

Create `tests/test_query_transform.py`:

```python
from app.rag.query_transform import QueryTransformer


class FakeLLM:
    def complete(self, prompt: str) -> str:
        if "不同角度改写" in prompt:
            return "RAG 核心步骤有哪些\n知识库问答系统流程"
        if "假设性回答" in prompt:
            return "RAG 系统通常包含文档解析、分块、向量化、检索和生成。"
        return ""


def test_query_transformer_generates_rewrite_and_hyde_variants():
    transformer = QueryTransformer(
        llm=FakeLLM(),
        rewrite_enabled=True,
        hyde_enabled=True,
        max_variants=4,
    )

    variants = transformer.expand("RAG 怎么做？")

    assert variants == [
        "RAG 怎么做？",
        "RAG 核心步骤有哪些",
        "知识库问答系统流程",
        "RAG 系统通常包含文档解析、分块、向量化、检索和生成。",
    ]


def test_query_transformer_can_disable_llm_expansion():
    transformer = QueryTransformer(
        llm=FakeLLM(),
        rewrite_enabled=False,
        hyde_enabled=False,
        max_variants=4,
    )

    assert transformer.expand("RAG 怎么做？") == ["RAG 怎么做？"]
```

- [ ] **Step 2: Run query transform tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_query_transform.py -v
```

Expected: FAIL because `app.rag.query_transform` does not exist.

- [ ] **Step 3: Implement query transformer**

Create `app/rag/query_transform.py`:

```python
class QueryTransformer:
    def __init__(self, llm, rewrite_enabled: bool, hyde_enabled: bool, max_variants: int):
        self.llm = llm
        self.rewrite_enabled = rewrite_enabled
        self.hyde_enabled = hyde_enabled
        self.max_variants = max(1, max_variants)

    def expand(self, query: str) -> list[str]:
        variants = [query]

        if self.rewrite_enabled and len(variants) < self.max_variants:
            variants.extend(self._rewrite(query))

        if self.hyde_enabled and len(variants) < self.max_variants:
            variants.append(self._hyde(query))

        cleaned: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            normalized = variant.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
            if len(cleaned) >= self.max_variants:
                break
        return cleaned

    def _rewrite(self, query: str) -> list[str]:
        prompt = (
            "请将以下用户问题从不同角度改写，生成2个语义相近但表述不同的查询。\n"
            "每行一个查询，不要编号。\n\n"
            f"原始问题：{query}"
        )
        response = self.llm.complete(prompt)
        return [line.strip(" -0123456789.、") for line in response.splitlines() if line.strip()]

    def _hyde(self, query: str) -> str:
        prompt = (
            "请针对以下问题，写一段简短的假设性回答。"
            "不需要保证准确性，只需要包含相关术语和概念，用于检索。\n\n"
            f"问题：{query}"
        )
        return self.llm.complete(prompt).strip()
```

- [ ] **Step 4: Add query expansion support to hybrid retriever**

Modify `app/rag/hybrid_retriever.py` constructor:

```python
        query_transformer=None,
```

Set:

```python
        self.query_transformer = query_transformer
```

At the start of `similarity_search`:

```python
        queries = self.query_transformer.expand(query) if self.query_transformer is not None else [query]
```

Replace single dense/sparse retrieval with:

```python
        ranked_lists = []
        for expanded_query in queries:
            dense_chunks = self.dense_retriever.similarity_search(expanded_query, top_k=self.dense_top_k)
            dense_documents = [chunk_to_retrieved_document(chunk) for chunk in dense_chunks]
            sparse_documents: list[RetrievedDocument] = self.sparse_retriever.search(expanded_query, top_k=self.sparse_top_k)
            ranked_lists.extend([dense_documents, sparse_documents])

        fused = reciprocal_rank_fusion(
            ranked_lists,
            top_k=max(top_k, self.reranker_top_n),
            k=self.rrf_k,
        )
```

Keep final reranking against the original user query:

```python
        reranked = self.reranker.rerank(query, fused, top_n=max(top_k, self.reranker_top_n))
```

- [ ] **Step 5: Wire query transformer in API**

Modify `app/api/routes.py` import:

```python
from app.rag.query_transform import QueryTransformer
```

In `build_retriever`, create a lightweight LLM for query transformation:

```python
    transformer_llm = OpenAIChatLLM(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
    )
    query_transformer = QueryTransformer(
        llm=transformer_llm,
        rewrite_enabled=settings.query_rewrite_enabled,
        hyde_enabled=settings.hyde_enabled,
        max_variants=settings.max_query_variants,
    )
```

Pass into `HybridRetriever`:

```python
        query_transformer=query_transformer,
```

- [ ] **Step 6: Run query and retrieval tests**

Run:

```bash
.venv/bin/pytest tests/test_query_transform.py tests/test_hybrid_retriever.py tests/test_api_retriever_factory.py -v
```

Expected: PASS. If existing fake API factory settings need `openai_api_key`, `openai_base_url`, or `chat_model`, add them to the test `SimpleNamespace` with harmless fake values.

- [ ] **Step 7: Commit**

Run:

```bash
git add app/rag/query_transform.py app/rag/hybrid_retriever.py app/api/routes.py tests/test_query_transform.py tests/test_hybrid_retriever.py tests/test_api_retriever_factory.py
git commit -m "feat: add query rewrite and hyde retrieval"
```

## Task 6: Stronger Evaluation Metrics And Report

**Files:**
- Modify: `app/evaluation/dataset.py`
- Modify: `app/evaluation/metrics.py`
- Create: `app/evaluation/report.py`
- Modify: `scripts/evaluate.py`
- Test: `tests/test_advanced_evaluation.py`

- [ ] **Step 1: Write failing advanced evaluation tests**

Create `tests/test_advanced_evaluation.py`:

```python
import json
from pathlib import Path

from app.evaluation.dataset import load_eval_cases
from app.evaluation.metrics import hit_rate_at_k, mean_reciprocal_rank, source_recall
from app.evaluation.report import build_retrieval_report
from app.rag.documents import RetrievedDocument


def docs(*sources):
    return [
        RetrievedDocument(id=str(index), content=source, source=source, metadata={})
        for index, source in enumerate(sources)
    ]


def test_eval_dataset_loads_expected_answer_keywords(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "RAG 是什么？",
                "ground_truth": "检索增强生成",
                "expected_sources": ["rag.md"],
                "expected_answer_keywords": ["检索", "生成"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    case = load_eval_cases(path)[0]

    assert case.expected_answer_keywords == ["检索", "生成"]


def test_retrieval_metrics_calculate_hit_mrr_and_recall():
    retrieved = [docs("a.md", "b.md"), docs("x.md", "y.md")]
    expected = [["b.md"], ["missing.md", "y.md"]]

    assert hit_rate_at_k(retrieved, expected) == 1.0
    assert mean_reciprocal_rank(retrieved, expected) == 0.5
    assert source_recall(retrieved, expected) == 0.5


def test_build_retrieval_report_returns_case_breakdown():
    cases = [
        {
            "question": "问题 A",
            "expected_sources": ["a.md"],
            "retrieved": docs("a.md"),
        }
    ]

    report = build_retrieval_report(cases)

    assert report["summary"]["cases"] == 1
    assert report["summary"]["hit_rate_at_k"] == 1.0
    assert report["cases"][0]["hit"] is True
```

- [ ] **Step 2: Run advanced evaluation tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_advanced_evaluation.py -v
```

Expected: FAIL because new fields/metrics/report do not exist.

- [ ] **Step 3: Extend evaluation dataset**

Modify `app/evaluation/dataset.py`:

```python
@dataclass(frozen=True)
class EvalCase:
    question: str
    ground_truth: str
    expected_sources: list[str]
    expected_answer_keywords: list[str]
```

Update loader:

```python
                expected_answer_keywords=[str(keyword) for keyword in row.get("expected_answer_keywords", [])],
```

- [ ] **Step 4: Implement MRR and source recall**

Modify `app/evaluation/metrics.py`:

```python
def mean_reciprocal_rank(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    if not expected_sources:
        return 0.0

    total = 0.0
    for documents, expected in zip(retrieved, expected_sources, strict=False):
        expected_set = set(expected)
        reciprocal = 0.0
        for rank, document in enumerate(documents, start=1):
            if document.source in expected_set:
                reciprocal = 1.0 / rank
                break
        total += reciprocal
    return total / len(expected_sources)


def source_recall(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    expected_total = sum(len(set(sources)) for sources in expected_sources)
    if expected_total == 0:
        return 0.0

    found = 0
    for documents, expected in zip(retrieved, expected_sources, strict=False):
        retrieved_sources = {document.source for document in documents}
        found += len(retrieved_sources & set(expected))
    return found / expected_total
```

- [ ] **Step 5: Implement report builder**

Create `app/evaluation/report.py`:

```python
from app.evaluation.metrics import hit_rate_at_k, mean_reciprocal_rank, source_recall


def build_retrieval_report(cases: list[dict]) -> dict:
    retrieved = [case["retrieved"] for case in cases]
    expected_sources = [case["expected_sources"] for case in cases]

    case_rows = []
    for case in cases:
        retrieved_sources = [document.source for document in case["retrieved"]]
        expected = set(case["expected_sources"])
        hit = bool(set(retrieved_sources) & expected)
        case_rows.append(
            {
                "question": case["question"],
                "expected_sources": case["expected_sources"],
                "retrieved_sources": retrieved_sources,
                "hit": hit,
            }
        )

    return {
        "summary": {
            "cases": len(cases),
            "hit_rate_at_k": hit_rate_at_k(retrieved, expected_sources),
            "mrr_at_k": mean_reciprocal_rank(retrieved, expected_sources),
            "source_recall": source_recall(retrieved, expected_sources),
        },
        "cases": case_rows,
    }
```

- [ ] **Step 6: Update evaluation script**

Modify `scripts/evaluate.py`:

```python
from app.evaluation.report import build_retrieval_report
```

Replace summary printing with:

```python
    report_cases = []
    for case, retrieved_documents in zip(cases, all_retrieved, strict=False):
        report_cases.append(
            {
                "question": case.question,
                "expected_sources": case.expected_sources,
                "retrieved": retrieved_documents,
            }
        )

    print(build_retrieval_report(report_cases))
```

- [ ] **Step 7: Update sample evaluation data**

Modify each row in `data/eval/sample_eval.jsonl` to include `expected_answer_keywords`:

```jsonl
{"question":"RAG 系统包含哪些核心步骤？","ground_truth":"文档解析、分块、向量化、检索、提示词组装和大模型回答生成。","expected_sources":["data/documents/example.md"],"expected_answer_keywords":["文档解析","向量化","检索","生成"]}
{"question":"系统如何保证答案可追溯？","ground_truth":"接口会返回引用来源，包括文件路径、页码和 chunk 编号。","expected_sources":["data/documents/example.md"],"expected_answer_keywords":["引用来源","文件路径","chunk"]}
```

- [ ] **Step 8: Run evaluation tests**

Run:

```bash
.venv/bin/pytest tests/test_advanced_evaluation.py tests/test_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add app/evaluation scripts/evaluate.py data/eval/sample_eval.jsonl tests/test_advanced_evaluation.py tests/test_evaluation.py
git commit -m "feat: add advanced retrieval evaluation report"
```

## Task 7: Evaluation Report API And Frontend Panel

**Files:**
- Modify: `app/api/schemas.py`
- Modify: `app/api/routes.py`
- Modify: `app/web/static/index.html`
- Modify: `app/web/static/app.js`
- Modify: `app/web/static/styles.css`
- Modify: `app/web/static/ui-utils.js`
- Modify: `tests/js/ui-utils.test.mjs`
- Test: `tests/test_evaluation_endpoint.py`

- [ ] **Step 1: Write failing evaluation endpoint test**

Create `tests/test_evaluation_endpoint.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_evaluation_report_endpoint(monkeypatch):
    from app.api import routes

    monkeypatch.setattr(
        routes,
        "build_evaluation_report",
        lambda: {"summary": {"cases": 1, "hit_rate_at_k": 1.0, "mrr_at_k": 1.0, "source_recall": 1.0}, "cases": []},
    )

    client = TestClient(app)
    response = client.get("/evaluation/report")

    assert response.status_code == 200
    assert response.json()["summary"]["hit_rate_at_k"] == 1.0
```

- [ ] **Step 2: Run endpoint test and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_evaluation_endpoint.py -v
```

Expected: FAIL because `/evaluation/report` does not exist.

- [ ] **Step 3: Add API report helper**

Modify `app/api/routes.py` imports:

```python
from app.evaluation.dataset import load_eval_cases
from app.evaluation.report import build_retrieval_report
from app.rag.documents import chunk_to_retrieved_document
```

Add helper:

```python
def build_evaluation_report():
    settings = get_settings()
    cases = load_eval_cases(settings.eval_dataset_path)
    retriever = build_retriever()

    report_cases = []
    for case in cases:
        chunks = retriever.similarity_search(case.question, top_k=settings.retrieval_top_k)
        report_cases.append(
            {
                "question": case.question,
                "expected_sources": case.expected_sources,
                "retrieved": [chunk_to_retrieved_document(chunk) for chunk in chunks],
            }
        )
    return build_retrieval_report(report_cases)
```

Add route:

```python
@router.get("/evaluation/report")
def evaluation_report():
    return build_evaluation_report()
```

- [ ] **Step 4: Add frontend utility formatter test**

Modify `tests/js/ui-utils.test.mjs`:

```javascript
import { formatPercent } from "../../app/web/static/ui-utils.js";

test("formatPercent formats ratio values", () => {
  assert.equal(formatPercent(1), "100.0%");
  assert.equal(formatPercent(0.825), "82.5%");
});
```

- [ ] **Step 5: Implement percent formatter**

Modify `app/web/static/ui-utils.js`:

```javascript
export function formatPercent(value) {
  const number = Number(value);
  if (Number.isNaN(number)) {
    return "0.0%";
  }
  return `${(number * 100).toFixed(1)}%`;
}
```

- [ ] **Step 6: Add evaluation panel to HTML**

Modify `app/web/static/index.html` inside the side panel after the warmup panel:

```html
          <section class="panel">
            <div class="panel-heading">
              <h2>评估</h2>
              <span id="evaluationState">未运行</span>
            </div>
            <p class="muted">使用真实检索链路计算 Hit Rate、MRR 和 Source Recall。</p>
            <button class="secondary-button" id="evaluationButton" type="button">运行评估</button>
            <div class="metric-grid" id="evaluationMetrics">
              <span>等待评估</span>
            </div>
          </section>
```

- [ ] **Step 7: Add evaluation UI behavior**

Modify `app/web/static/app.js` import:

```javascript
import { clampTopK, escapeHtml, formatPercent, normalizeSources, parseSseChunk } from "./ui-utils.js";
```

Add elements:

```javascript
  evaluationButton: document.querySelector("#evaluationButton"),
  evaluationState: document.querySelector("#evaluationState"),
  evaluationMetrics: document.querySelector("#evaluationMetrics"),
```

Add function:

```javascript
async function runEvaluation() {
  elements.evaluationButton.disabled = true;
  setStatus(elements.evaluationState, "运行中");
  elements.evaluationMetrics.innerHTML = "<span>正在计算...</span>";

  try {
    const response = await fetch("/evaluation/report");
    const payload = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(JSON.stringify(payload));
    }
    const summary = payload.summary ?? {};
    elements.evaluationMetrics.innerHTML = `
      <div><strong>${escapeHtml(summary.cases ?? 0)}</strong><span>Cases</span></div>
      <div><strong>${formatPercent(summary.hit_rate_at_k)}</strong><span>Hit@K</span></div>
      <div><strong>${formatPercent(summary.mrr_at_k)}</strong><span>MRR@K</span></div>
      <div><strong>${formatPercent(summary.source_recall)}</strong><span>Source Recall</span></div>
    `;
    setStatus(elements.evaluationState, "完成", "is-ok");
  } catch (error) {
    setStatus(elements.evaluationState, "失败", "is-error");
    elements.evaluationMetrics.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    elements.evaluationButton.disabled = false;
  }
}
```

Register event:

```javascript
elements.evaluationButton.addEventListener("click", runEvaluation);
```

- [ ] **Step 8: Add metric grid CSS**

Modify `app/web/static/styles.css`:

```css
.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.metric-grid div {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface-soft);
  padding: 10px;
}

.metric-grid strong {
  display: block;
  font-size: 20px;
}

.metric-grid span {
  color: var(--text-muted);
  font-size: 12px;
}
```

- [ ] **Step 9: Run endpoint and frontend tests**

Run:

```bash
.venv/bin/pytest tests/test_evaluation_endpoint.py tests/test_web_static.py -v
node --test tests/js/ui-utils.test.mjs
```

Expected: PASS.

- [ ] **Step 10: Commit**

Run:

```bash
git add app/api app/web/static tests/test_evaluation_endpoint.py tests/js/ui-utils.test.mjs
git commit -m "feat: show retrieval evaluation in frontend"
```

## Task 8: Documentation And Final Verification

**Files:**
- Modify: `README.md`

- [x] **Step 1: Update README with Phase 4 features**

Add a `第四阶段能力` section:

```markdown
## 第四阶段能力

- 建立索引时生成父块语料，检索命中子块后返回父块上下文。
- 支持 Query 改写和 HyDE 查询扩展，默认最多生成 4 个查询变体。
- 评估脚本输出 Hit Rate@K、MRR@K 和 Source Recall。
- 前端可以直接运行评估并查看指标。
```

Update commands:

```bash
python -m scripts.ingest
python -m scripts.evaluate
python -m scripts.warmup
uvicorn app.main:app --reload
```

- [x] **Step 2: Run full Python tests**

Run:

```bash
.venv/bin/pytest -v
```

Expected: PASS.

- [x] **Step 3: Run JavaScript tests**

Run:

```bash
node --test tests/js/ui-utils.test.mjs
```

Expected: PASS.

- [x] **Step 4: Run ingestion**

Run:

```bash
.venv/bin/python -m scripts.ingest
```

Expected: output contains `IngestResult(...)` and `indexed_chunks` greater than `0`.

- [x] **Step 5: Run evaluation**

Run:

```bash
.venv/bin/python -m scripts.evaluate
```

Expected: output contains `summary`, `hit_rate_at_k`, `mrr_at_k`, and `source_recall`.

- [x] **Step 6: Run warmup**

Run:

```bash
.venv/bin/python -m scripts.warmup
```

Expected: output contains `{"retriever": "HybridRetriever", "reranker": "ScoreBasedReranker", "status": "ok"}`.

- [x] **Step 7: Verify API routes**

Run:

```bash
.venv/bin/python -c "from app.main import app; print(sorted(route.path for route in app.routes if route.path in ['/', '/health', '/ingest', '/ask', '/ask/stream', '/evaluation/report']))"
```

Expected:

```text
['/', '/ask', '/ask/stream', '/evaluation/report', '/health', '/ingest']
```

- [x] **Step 8: Browser verification**

Start server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

Verify:

- Page loads.
- Health becomes online.
- Rebuild index works.
- Running evaluation shows metrics.
- Asking a question streams an answer and shows sources.
- Layout remains readable on mobile width.

- [x] **Step 9: Stop server**

Stop the `uvicorn` process with `Ctrl+C`.

- [x] **Step 10: Commit docs and push**

Run:

```bash
git add README.md docs/superpowers/plans/2026-06-14-rag-knowledge-qa-phase-4-advanced-rag.md
git commit -m "docs: plan phase 4 advanced rag"
git push
```

## Plan Self-Review

- Spec coverage: This plan covers the reference project's remaining local RAG depth: parent-child indexing, query rewrite, HyDE, stronger evaluation metrics, and UI-visible evaluation.
- Scope control: It leaves full RAGAS, Milvus, Redis, K8s, and multimodal OCR/VLM processing to later infrastructure-heavy phases.
- Placeholder scan: Each implementation task includes concrete files, code, tests, commands, and expected results.
- Type consistency: `Chunk`, `RetrievedDocument`, `HybridRetriever`, `JsonlParentStore`, `QueryTransformer`, `EvalCase`, and report field names are consistent across tasks.
- Verification path: The plan ends with full Python tests, Node tests, ingestion, evaluation, warmup, route verification, browser verification, commit, and push.
