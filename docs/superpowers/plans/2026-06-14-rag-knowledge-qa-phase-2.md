# RAG Knowledge QA Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the MVP RAG system from single dense retrieval to an interview-ready second phase with richer document parsing, hybrid retrieval, mandatory reranking, RAGAS-style offline evaluation scaffolding, and SSE streaming responses.

**Architecture:** Keep the current FastAPI + Chroma + local BGE embedding MVP intact, then add retrieval components behind narrow interfaces. Ingestion will preserve a plain-text corpus for BM25, retrieval will combine dense Chroma results and BM25 results through RRF, reranking will be a required step in the retrieval chain, and evaluation/streaming will sit at the service/API edge without rewriting the ingestion pipeline.

**Tech Stack:** Python 3.11, FastAPI, Chroma, sentence-transformers, transformers 4.x, rank-bm25, jieba, python-docx, beautifulsoup4, sse-starlette, pytest, FlagEmbedding, and Phase 7 onward required RAGAS/datasets for answer-level automated evaluation.

---

## Scope

### In Scope

- Add `.docx`, `.html`, and `.htm` document loading.
- Add Chinese-aware BM25 sparse retrieval over indexed chunks.
- Add Reciprocal Rank Fusion (RRF) to merge dense Chroma and BM25 results.
- Add required local BGE reranker integration with deterministic adapter tests.
- Add a hybrid retrieval service that preserves the existing `similarity_search(query, top_k)` interface.
- Add SSE streaming endpoint for answer text.
- Add lightweight evaluation dataset format and offline evaluation script.
- Add tests for loaders, BM25, RRF, hybrid retrieval, reranker adapter behavior, and streaming response formatting.
- Update README with Phase 2 capabilities and usage.

### Out of Scope

- Milvus migration.
- Redis cache.
- Kubernetes deployment.
- Full production observability stack.
- Image OCR and VLM-based chart understanding.
- Real RAGAS score guarantees from a 200-question labeled dataset.

These are valuable, but they are third-stage work. Phase 2 should produce a stronger local project without swallowing the whole ocean. Tiny ocean snack only.

## File Structure

- Modify: `pyproject.toml` to add Phase 2 dependencies.
- Modify: `.env.example` to add retrieval and reranking settings.
- Modify: `README.md` to document Phase 2 usage.
- Modify: `app/core/config.py` to expose new settings.
- Modify: `app/ingestion/loaders.py` to support `.docx`, `.html`, `.htm`.
- Modify: `app/ingestion/pipeline.py` to write a persisted BM25 corpus file after chunking.
- Create: `app/rag/documents.py` for stable chunk identity and source conversion helpers.
- Create: `app/rag/bm25.py` for sparse retrieval.
- Create: `app/rag/fusion.py` for RRF.
- Create: `app/rag/reranker.py` for BGE reranking.
- Create: `app/rag/hybrid_retriever.py` for dense + sparse + mandatory rerank retrieval.
- Modify: `app/rag/vector_store.py` to use stable chunk IDs and include `chunk_id` metadata.
- Modify: `app/rag/llm.py` to expose streaming completion.
- Modify: `app/rag/service.py` to expose `answer_stream`.
- Modify: `app/api/schemas.py` if endpoint-specific schema additions are needed.
- Modify: `app/api/routes.py` to build hybrid retriever and expose `/ask/stream`.
- Create: `app/evaluation/__init__.py` package marker.
- Create: `app/evaluation/dataset.py` for JSONL test-case loading.
- Create: `app/evaluation/metrics.py` for lightweight local retrieval metrics.
- Create: `scripts/evaluate.py` offline evaluation entry point.
- Create: `scripts/warmup.py` model warmup entry point.
- Create: `data/eval/sample_eval.jsonl` sample evaluation set.
- Create: `tests/test_docx_html_loaders.py`.
- Create: `tests/test_bm25.py`.
- Create: `tests/test_fusion.py`.
- Create: `tests/test_hybrid_retriever.py`.
- Create: `tests/test_reranker.py`.
- Create: `tests/test_evaluation.py`.
- Create: `tests/test_streaming.py`.
- Create: `tests/test_warmup.py`.

## Task 1: Phase 2 Dependencies And Settings

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `app/core/config.py`
- Test: `tests/test_phase2_config.py`

- [ ] **Step 1: Write failing config test**

Create `tests/test_phase2_config.py`:

```python
from app.core.config import Settings


def test_phase2_settings_defaults():
    settings = Settings()

    assert settings.bm25_corpus_path.as_posix() == "data/chroma/bm25_corpus.jsonl"
    assert settings.dense_retrieval_top_k == 20
    assert settings.bm25_retrieval_top_k == 20
    assert settings.rrf_k == 60
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker_top_n == 5
```

- [ ] **Step 2: Run config test and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_phase2_config.py -v
```

Expected: FAIL because the new settings do not exist.

- [ ] **Step 3: Add dependencies**

Modify `pyproject.toml` dependencies so the list includes these new packages:

```toml
    "beautifulsoup4>=4.12.3",
    "jieba>=0.42.1",
    "python-docx>=1.1.2",
    "rank-bm25>=0.2.2",
    "sse-starlette>=2.1.0",
```

Keep existing dependencies unchanged.

- [ ] **Step 4: Add environment settings**

Append to `.env.example`:

```dotenv
BM25_CORPUS_PATH=data/chroma/bm25_corpus.jsonl
DENSE_RETRIEVAL_TOP_K=20
BM25_RETRIEVAL_TOP_K=20
RRF_K=60
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_TOP_N=5
```

- [ ] **Step 5: Update settings model**

Modify `app/core/config.py` by adding these fields to `Settings`:

```python
    bm25_corpus_path: Path = Field(default=Path("data/chroma/bm25_corpus.jsonl"), alias="BM25_CORPUS_PATH")
    dense_retrieval_top_k: int = Field(default=20, alias="DENSE_RETRIEVAL_TOP_K")
    bm25_retrieval_top_k: int = Field(default=20, alias="BM25_RETRIEVAL_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    reranker_top_n: int = Field(default=5, alias="RERANKER_TOP_N")
```

- [ ] **Step 6: Run config test and verify pass**

Run:

```bash
.venv/bin/pytest tests/test_phase2_config.py -v
```

Expected: PASS.

- [ ] **Step 7: Install updated dependencies**

Run:

```bash
.venv/bin/pip install -e ".[dev]"
```

Expected: completes without dependency resolution errors.

- [ ] **Step 8: Commit**

Run:

```bash
git add pyproject.toml .env.example app/core/config.py tests/test_phase2_config.py
git commit -m "chore: add phase 2 retrieval settings"
```

## Task 2: DOCX And HTML Document Loading

**Files:**
- Modify: `app/ingestion/loaders.py`
- Test: `tests/test_docx_html_loaders.py`

- [ ] **Step 1: Write failing loader tests**

Create `tests/test_docx_html_loaders.py`:

```python
from pathlib import Path

from docx import Document as DocxDocument

from app.ingestion.loaders import load_document


def test_load_docx_document(tmp_path: Path):
    path = tmp_path / "policy.docx"
    doc = DocxDocument()
    doc.add_heading("制度文档", level=1)
    doc.add_paragraph("员工可以通过知识库查询规章制度。")
    doc.save(path)

    loaded = load_document(path)

    assert len(loaded) == 1
    assert "制度文档" in loaded[0].text
    assert "员工可以通过知识库查询规章制度。" in loaded[0].text
    assert loaded[0].metadata["file_type"] == ".docx"


def test_load_html_document_removes_scripts_and_keeps_text(tmp_path: Path):
    path = tmp_path / "guide.html"
    path.write_text(
        """
        <html>
          <head><script>console.log("ignore me")</script></head>
          <body>
            <h1>系统指南</h1>
            <p>RAG 系统支持知识库问答。</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    loaded = load_document(path)

    assert len(loaded) == 1
    assert "系统指南" in loaded[0].text
    assert "RAG 系统支持知识库问答。" in loaded[0].text
    assert "console.log" not in loaded[0].text
    assert loaded[0].metadata["file_type"] == ".html"
```

- [ ] **Step 2: Run new loader tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_docx_html_loaders.py -v
```

Expected: FAIL because `.docx` and `.html` are unsupported.

- [ ] **Step 3: Implement DOCX and HTML loaders**

Modify `app/ingestion/loaders.py`:

```python
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
```

Change supported suffixes:

```python
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}
```

Add these branches to `load_document` after the `.txt` / `.md` branch and before PDF handling:

```python
    if suffix == ".docx":
        return _load_docx(path)

    if suffix in {".html", ".htm"}:
        return _load_html(path)
```

Add helper functions:

```python
def _load_docx(path: Path) -> list[LoadedDocument]:
    doc = DocxDocument(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text:
        return []
    return [
        LoadedDocument(
            text=text,
            source=str(path),
            metadata={"file_type": ".docx"},
        )
    ]


def _load_html(path: Path) -> list[LoadedDocument]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = "\n".join(lines)
    if not normalized:
        return []
    return [
        LoadedDocument(
            text=normalized,
            source=str(path),
            metadata={"file_type": path.suffix.lower()},
        )
    ]
```

- [ ] **Step 4: Run loader tests and existing loader tests**

Run:

```bash
.venv/bin/pytest tests/test_docx_html_loaders.py tests/test_loaders.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add app/ingestion/loaders.py tests/test_docx_html_loaders.py
git commit -m "feat: support docx and html ingestion"
```

## Task 3: Stable Chunk Documents And BM25 Corpus Persistence

**Files:**
- Create: `app/rag/documents.py`
- Modify: `app/rag/vector_store.py`
- Modify: `app/ingestion/pipeline.py`
- Test: `tests/test_documents.py`
- Test: `tests/test_pipeline_corpus.py`

- [ ] **Step 1: Write failing document helper tests**

Create `tests/test_documents.py`:

```python
from app.ingestion.chunker import Chunk
from app.rag.documents import RetrievedDocument, chunk_id, chunk_to_retrieved_document


def test_chunk_id_is_stable_for_same_source_and_index():
    chunk = Chunk(
        content="same content",
        source="data/documents/example.md",
        metadata={"chunk_index": 3, "source": "data/documents/example.md"},
    )

    assert chunk_id(chunk) == chunk_id(chunk)


def test_chunk_to_retrieved_document_sets_identity():
    chunk = Chunk(
        content="RAG 知识",
        source="data/documents/example.md",
        metadata={"chunk_index": 1, "source": "data/documents/example.md", "page": 2},
    )

    doc = chunk_to_retrieved_document(chunk, score=0.42)

    assert isinstance(doc, RetrievedDocument)
    assert doc.id == chunk_id(chunk)
    assert doc.content == "RAG 知识"
    assert doc.source == "data/documents/example.md"
    assert doc.metadata["page"] == 2
    assert doc.score == 0.42
```

- [ ] **Step 2: Write failing corpus persistence test**

Create `tests/test_pipeline_corpus.py`:

```python
import json
from pathlib import Path

from app.ingestion.pipeline import persist_bm25_corpus
from app.ingestion.chunker import Chunk


def test_persist_bm25_corpus_writes_jsonl(tmp_path: Path):
    corpus_path = tmp_path / "bm25.jsonl"
    chunks = [
        Chunk(
            content="RAG 系统包含检索和生成。",
            source="example.md",
            metadata={"source": "example.md", "chunk_index": 0, "file_type": ".md"},
        )
    ]

    persist_bm25_corpus(chunks, corpus_path)

    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "id": rows[0]["id"],
            "content": "RAG 系统包含检索和生成。",
            "source": "example.md",
            "metadata": {"source": "example.md", "chunk_index": 0, "file_type": ".md", "chunk_id": rows[0]["id"]},
        }
    ]
    assert rows[0]["id"]
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_documents.py tests/test_pipeline_corpus.py -v
```

Expected: FAIL because helper module and `persist_bm25_corpus` do not exist.

- [ ] **Step 4: Implement document helper**

Create `app/rag/documents.py`:

```python
from dataclasses import dataclass
from hashlib import sha1

from app.ingestion.chunker import Chunk


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    content: str
    source: str
    metadata: dict
    score: float | None = None


def chunk_id(chunk: Chunk) -> str:
    chunk_index = chunk.metadata.get("chunk_index", "")
    raw = f"{chunk.source}:{chunk_index}:{chunk.content}"
    return sha1(raw.encode("utf-8")).hexdigest()


def chunk_to_retrieved_document(chunk: Chunk, score: float | None = None) -> RetrievedDocument:
    metadata = dict(chunk.metadata)
    identity = str(metadata.get("chunk_id") or chunk_id(chunk))
    metadata["chunk_id"] = identity
    metadata["source"] = chunk.source
    return RetrievedDocument(
        id=identity,
        content=chunk.content,
        source=chunk.source,
        metadata=metadata,
        score=score,
    )


def retrieved_document_to_chunk(document: RetrievedDocument) -> Chunk:
    metadata = dict(document.metadata)
    metadata["chunk_id"] = document.id
    metadata["source"] = document.source
    return Chunk(content=document.content, source=document.source, metadata=metadata)
```

- [ ] **Step 5: Update vector store to persist stable IDs**

Modify `app/rag/vector_store.py`:

```python
from app.rag.documents import chunk_id
```

In `add_chunks`, replace ID and metadata creation:

```python
        ids = [chunk_id(chunk) for chunk in chunks]
        metadatas = []
        for chunk, identity in zip(chunks, ids, strict=False):
            metadata = dict(chunk.metadata)
            metadata["chunk_id"] = identity
            metadata["source"] = chunk.source
            metadatas.append(metadata)
```

- [ ] **Step 6: Add BM25 corpus persistence to pipeline**

Modify `app/ingestion/pipeline.py`:

```python
import json
```

Add import:

```python
from app.rag.documents import chunk_id
```

Add function:

```python
def persist_bm25_corpus(chunks, corpus_path: Path) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            identity = chunk_id(chunk)
            metadata = dict(chunk.metadata)
            metadata["chunk_id"] = identity
            metadata["source"] = chunk.source
            row = {
                "id": identity,
                "content": chunk.content,
                "source": chunk.source,
                "metadata": metadata,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Change `ingest_directory` signature:

```python
    bm25_corpus_path: Path | None = None,
```

After chunking and before return:

```python
    if bm25_corpus_path is not None:
        persist_bm25_corpus(chunks, bm25_corpus_path)
```

- [ ] **Step 7: Pass BM25 corpus path from API and script**

Modify `scripts/ingest.py` ingestion call:

```python
        bm25_corpus_path=settings.bm25_corpus_path,
```

Modify `app/api/routes.py` ingestion call:

```python
        bm25_corpus_path=settings.bm25_corpus_path,
```

- [ ] **Step 8: Run document and pipeline tests**

Run:

```bash
.venv/bin/pytest tests/test_documents.py tests/test_pipeline_corpus.py tests/test_rag_service.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add app/rag/documents.py app/rag/vector_store.py app/ingestion/pipeline.py app/api/routes.py scripts/ingest.py tests/test_documents.py tests/test_pipeline_corpus.py
git commit -m "feat: persist bm25 retrieval corpus"
```

## Task 4: BM25 Sparse Retriever

**Files:**
- Create: `app/rag/bm25.py`
- Test: `tests/test_bm25.py`

- [ ] **Step 1: Write failing BM25 tests**

Create `tests/test_bm25.py`:

```python
import json
from pathlib import Path

from app.rag.bm25 import BM25Retriever, tokenize


def test_tokenize_handles_chinese_and_ascii():
    tokens = tokenize("RAG 系统支持 BM25 检索")

    assert "rag" in tokens
    assert "bm25" in tokens
    assert "检索" in tokens


def test_bm25_retriever_returns_ranked_documents(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {
            "id": "a",
            "content": "RAG 系统支持向量检索和关键词检索。",
            "source": "a.md",
            "metadata": {"source": "a.md", "chunk_index": 0, "chunk_id": "a"},
        },
        {
            "id": "b",
            "content": "员工报销流程需要提交发票。",
            "source": "b.md",
            "metadata": {"source": "b.md", "chunk_index": 0, "chunk_id": "b"},
        },
    ]
    corpus.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    retriever = BM25Retriever.from_jsonl(corpus)
    results = retriever.search("关键词检索", top_k=1)

    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].source == "a.md"
    assert results[0].score is not None
```

- [ ] **Step 2: Run BM25 tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_bm25.py -v
```

Expected: FAIL because `app.rag.bm25` does not exist.

- [ ] **Step 3: Implement BM25 retriever**

Create `app/rag/bm25.py`:

```python
import json
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from app.rag.documents import RetrievedDocument


ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    ascii_tokens = ASCII_TOKEN_RE.findall(lowered)
    chinese_tokens = [
        token.strip().lower()
        for token in jieba.lcut(text)
        if token.strip() and not token.isspace()
    ]
    tokens = ascii_tokens + chinese_tokens
    return [token for token in tokens if token]


class BM25Retriever:
    def __init__(self, documents: list[RetrievedDocument]):
        self.documents = documents
        self.tokenized_corpus = [tokenize(document.content) for document in documents]
        self.index = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    @classmethod
    def from_jsonl(cls, corpus_path: Path) -> "BM25Retriever":
        if not corpus_path.exists():
            return cls([])

        documents: list[RetrievedDocument] = []
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            documents.append(
                RetrievedDocument(
                    id=str(row["id"]),
                    content=str(row["content"]),
                    source=str(row["source"]),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
        return cls(documents)

    def search(self, query: str, top_k: int) -> list[RetrievedDocument]:
        if self.index is None or not self.documents:
            return []

        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[RetrievedDocument] = []
        for index, score in ranked[:top_k]:
            if score <= 0:
                continue
            document = self.documents[index]
            results.append(
                RetrievedDocument(
                    id=document.id,
                    content=document.content,
                    source=document.source,
                    metadata=dict(document.metadata),
                    score=float(score),
                )
            )
        return results
```

- [ ] **Step 4: Run BM25 tests and verify pass**

Run:

```bash
.venv/bin/pytest tests/test_bm25.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add app/rag/bm25.py tests/test_bm25.py
git commit -m "feat: add bm25 sparse retriever"
```

## Task 5: RRF Fusion

**Files:**
- Create: `app/rag/fusion.py`
- Test: `tests/test_fusion.py`

- [ ] **Step 1: Write failing RRF tests**

Create `tests/test_fusion.py`:

```python
from app.rag.documents import RetrievedDocument
from app.rag.fusion import reciprocal_rank_fusion


def doc(identity: str, content: str = "content") -> RetrievedDocument:
    return RetrievedDocument(id=identity, content=content, source=f"{identity}.md", metadata={"chunk_id": identity})


def test_rrf_deduplicates_and_prefers_documents_ranked_by_multiple_retrievers():
    dense = [doc("a"), doc("b")]
    sparse = [doc("b"), doc("c")]

    fused = reciprocal_rank_fusion([dense, sparse], top_k=3, k=60)

    assert [item.id for item in fused] == ["b", "a", "c"]
    assert fused[0].score is not None


def test_rrf_handles_empty_lists():
    fused = reciprocal_rank_fusion([[], [doc("x")]], top_k=2, k=60)

    assert [item.id for item in fused] == ["x"]
```

- [ ] **Step 2: Run RRF tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_fusion.py -v
```

Expected: FAIL because `app.rag.fusion` does not exist.

- [ ] **Step 3: Implement RRF**

Create `app/rag/fusion.py`:

```python
from app.rag.documents import RetrievedDocument


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDocument]],
    top_k: int,
    k: int = 60,
) -> list[RetrievedDocument]:
    scores: dict[str, float] = {}
    documents: dict[str, RetrievedDocument] = {}

    for ranked in ranked_lists:
        for rank, document in enumerate(ranked, start=1):
            documents.setdefault(document.id, document)
            scores[document.id] = scores.get(document.id, 0.0) + 1.0 / (k + rank)

    ordered_ids = sorted(scores, key=lambda identity: scores[identity], reverse=True)
    fused: list[RetrievedDocument] = []
    for identity in ordered_ids[:top_k]:
        document = documents[identity]
        fused.append(
            RetrievedDocument(
                id=document.id,
                content=document.content,
                source=document.source,
                metadata=dict(document.metadata),
                score=scores[identity],
            )
        )
    return fused
```

- [ ] **Step 4: Run RRF tests and verify pass**

Run:

```bash
.venv/bin/pytest tests/test_fusion.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add app/rag/fusion.py tests/test_fusion.py
git commit -m "feat: add rrf result fusion"
```

## Task 6: Required Reranker

**Files:**
- Create: `app/rag/reranker.py`
- Test: `tests/test_reranker.py`

- [ ] **Step 1: Write failing reranker tests**

Create `tests/test_reranker.py`:

```python
import sys
from types import ModuleType

from app.rag.documents import RetrievedDocument
from app.rag.reranker import ScoreBasedReranker, build_bge_reranker


def test_score_based_reranker_sorts_by_model_score():
    class FakeModel:
        def compute_score(self, pairs):
            assert pairs == [["query", "A"], ["query", "B"]]
            return [0.1, 0.9]

    docs = [
        RetrievedDocument(id="a", content="A", source="a.md", metadata={}),
        RetrievedDocument(id="b", content="B", source="b.md", metadata={}),
    ]

    reranker = ScoreBasedReranker(FakeModel())
    results = reranker.rerank("query", docs, top_n=2)

    assert [doc.id for doc in results] == ["b", "a"]
    assert results[0].score == 0.9


def test_build_bge_reranker_uses_required_model(monkeypatch):
    created = {}

    class FakeFlagReranker:
        def __init__(self, model_name: str, use_fp16: bool):
            created["model_name"] = model_name
            created["use_fp16"] = use_fp16

        def compute_score(self, pairs):
            return [1.0 for _ in pairs]

    fake_module = ModuleType("FlagEmbedding")
    fake_module.FlagReranker = FakeFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)

    reranker = build_bge_reranker("BAAI/bge-reranker-v2-m3")

    assert isinstance(reranker, ScoreBasedReranker)
    assert created == {"model_name": "BAAI/bge-reranker-v2-m3", "use_fp16": True}
```

- [ ] **Step 2: Run reranker tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_reranker.py -v
```

Expected: FAIL because `app.rag.reranker` does not exist.

- [ ] **Step 3: Implement reranker abstractions**

Create `app/rag/reranker.py`:

```python
from typing import Protocol

from app.rag.documents import RetrievedDocument


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[RetrievedDocument], top_n: int) -> list[RetrievedDocument]:
        ...


class ScoreBasedReranker:
    def __init__(self, model):
        self.model = model

    def rerank(self, query: str, documents: list[RetrievedDocument], top_n: int) -> list[RetrievedDocument]:
        if not documents:
            return []

        pairs = [[query, document.content] for document in documents]
        scores = self.model.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]

        scored = []
        for document, score in zip(documents, scores, strict=False):
            scored.append(
                RetrievedDocument(
                    id=document.id,
                    content=document.content,
                    source=document.source,
                    metadata=dict(document.metadata),
                    score=float(score),
                )
            )
        scored.sort(key=lambda document: document.score or 0.0, reverse=True)
        return scored[:top_n]


def build_bge_reranker(model_name: str):
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError(
            "BGE reranking is required and needs FlagEmbedding. "
            "Install it separately with `pip install FlagEmbedding`."
        ) from exc

    return ScoreBasedReranker(FlagReranker(model_name, use_fp16=True))
```

- [ ] **Step 4: Run reranker tests and verify pass**

Run:

```bash
.venv/bin/pytest tests/test_reranker.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add app/rag/reranker.py tests/test_reranker.py
git commit -m "feat: add required reranker adapter"
```

## Task 7: Hybrid Retriever

**Files:**
- Create: `app/rag/hybrid_retriever.py`
- Modify: `app/api/routes.py`
- Test: `tests/test_hybrid_retriever.py`

- [ ] **Step 1: Write failing hybrid retriever tests**

Create `tests/test_hybrid_retriever.py`:

```python
from app.ingestion.chunker import Chunk
from app.rag.documents import RetrievedDocument
from app.rag.hybrid_retriever import HybridRetriever


class FakeDenseRetriever:
    def similarity_search(self, query: str, top_k: int):
        assert query == "RAG 检索"
        assert top_k == 2
        return [
            Chunk(content="向量检索内容", source="dense.md", metadata={"source": "dense.md", "chunk_index": 0}),
            Chunk(content="重复内容", source="same.md", metadata={"source": "same.md", "chunk_index": 1}),
        ]


class FakeSparseRetriever:
    def search(self, query: str, top_k: int):
        assert query == "RAG 检索"
        assert top_k == 2
        return [
            RetrievedDocument(id="same", content="重复内容", source="same.md", metadata={"chunk_id": "same", "chunk_index": 1}),
            RetrievedDocument(id="sparse", content="关键词检索内容", source="sparse.md", metadata={"chunk_id": "sparse", "chunk_index": 0}),
        ]


class FakeReranker:
    def rerank(self, query, documents, top_n):
        assert query == "RAG 检索"
        assert top_n == 2
        return list(reversed(documents))[:top_n]


def test_hybrid_retriever_returns_chunks_after_fusion_and_rerank():
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        sparse_retriever=FakeSparseRetriever(),
        reranker=FakeReranker(),
        dense_top_k=2,
        sparse_top_k=2,
        rrf_k=60,
        reranker_top_n=2,
    )

    chunks = retriever.similarity_search("RAG 检索", top_k=2)

    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].content in {"向量检索内容", "关键词检索内容", "重复内容"}
```

- [ ] **Step 2: Run hybrid tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_hybrid_retriever.py -v
```

Expected: FAIL because `app.rag.hybrid_retriever` does not exist.

- [ ] **Step 3: Implement hybrid retriever**

Create `app/rag/hybrid_retriever.py`:

```python
from app.rag.documents import (
    RetrievedDocument,
    chunk_to_retrieved_document,
    retrieved_document_to_chunk,
)
from app.rag.fusion import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        reranker,
        dense_top_k: int,
        sparse_top_k: int,
        rrf_k: int,
        reranker_top_n: int,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.reranker_top_n = reranker_top_n

    def similarity_search(self, query: str, top_k: int):
        dense_chunks = self.dense_retriever.similarity_search(query, top_k=self.dense_top_k)
        dense_documents = [chunk_to_retrieved_document(chunk) for chunk in dense_chunks]
        sparse_documents: list[RetrievedDocument] = self.sparse_retriever.search(query, top_k=self.sparse_top_k)

        fused = reciprocal_rank_fusion([dense_documents, sparse_documents], top_k=max(top_k, self.reranker_top_n), k=self.rrf_k)
        reranked = self.reranker.rerank(query, fused, top_n=max(top_k, self.reranker_top_n))
        return [retrieved_document_to_chunk(document) for document in reranked[:top_k]]
```

- [ ] **Step 4: Wire hybrid retriever in API factory**

Modify `app/api/routes.py` imports:

```python
from app.rag.bm25 import BM25Retriever
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import build_bge_reranker
```

Add factory:

```python
def build_retriever():
    settings = get_settings()
    dense = build_vector_store()
    sparse = BM25Retriever.from_jsonl(settings.bm25_corpus_path)
    reranker = build_bge_reranker(settings.reranker_model)
    return HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
        dense_top_k=settings.dense_retrieval_top_k,
        sparse_top_k=settings.bm25_retrieval_top_k,
        rrf_k=settings.rrf_k,
        reranker_top_n=settings.reranker_top_n,
    )
```

Change `build_rag_service`:

```python
    return RagService(retriever=build_retriever(), llm=llm)
```

- [ ] **Step 5: Run hybrid tests and service tests**

Run:

```bash
.venv/bin/pytest tests/test_hybrid_retriever.py tests/test_rag_service.py -v
```

Expected: PASS.

- [ ] **Step 6: Run full tests**

Run:

```bash
.venv/bin/pytest -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add app/rag/hybrid_retriever.py app/api/routes.py tests/test_hybrid_retriever.py
git commit -m "feat: use hybrid rag retriever"
```

## Task 8: SSE Streaming Answer Endpoint

**Files:**
- Modify: `app/rag/llm.py`
- Modify: `app/rag/service.py`
- Modify: `app/api/routes.py`
- Test: `tests/test_streaming.py`

- [ ] **Step 1: Write failing streaming service test**

Create `tests/test_streaming.py`:

```python
from app.ingestion.chunker import Chunk
from app.rag.service import RagService


class FakeRetriever:
    def similarity_search(self, query: str, top_k: int):
        return [
            Chunk(
                content="RAG 系统支持流式输出。",
                source="stream.md",
                metadata={"source": "stream.md", "chunk_index": 0},
            )
        ]


class FakeStreamingLLM:
    def stream(self, prompt: str):
        yield "第一段"
        yield "第二段"

    def complete(self, prompt: str) -> str:
        return "第一段第二段"


def test_answer_stream_yields_text_chunks_then_sources_event():
    service = RagService(retriever=FakeRetriever(), llm=FakeStreamingLLM())

    events = list(service.answer_stream("怎么流式输出？", top_k=4))

    assert events[0] == {"event": "token", "data": "第一段"}
    assert events[1] == {"event": "token", "data": "第二段"}
    assert events[2]["event"] == "sources"
    assert "stream.md" in events[2]["data"]
```

- [ ] **Step 2: Run streaming test and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_streaming.py -v
```

Expected: FAIL because `answer_stream` does not exist.

- [ ] **Step 3: Add LLM stream method**

Modify `app/rag/llm.py`:

```python
    def stream(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
```

- [ ] **Step 4: Add service streaming**

Modify `app/rag/service.py`:

```python
import json
```

Add method to `RagService`:

```python
    def answer_stream(self, question: str, top_k: int):
        chunks = self.retriever.similarity_search(question, top_k=top_k)
        if not chunks:
            yield {"event": "token", "data": "知识库中没有找到相关内容，无法基于现有资料回答。"}
            yield {"event": "sources", "data": "[]"}
            return

        prompt = build_rag_prompt(question, chunks)
        stream = self.llm.stream(prompt) if hasattr(self.llm, "stream") else [self.llm.complete(prompt)]
        for token in stream:
            yield {"event": "token", "data": token}

        sources = [
            {
                "source": chunk.source,
                "page": chunk.metadata.get("page"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "content": chunk.content,
            }
            for chunk in chunks
        ]
        yield {"event": "sources", "data": json.dumps(sources, ensure_ascii=False)}
```

- [ ] **Step 5: Add `/ask/stream` endpoint**

Modify `app/api/routes.py` imports:

```python
from sse_starlette.sse import EventSourceResponse
```

Add endpoint:

```python
@router.post("/ask/stream")
def ask_stream(request: AskRequest):
    settings = get_settings()
    service = build_rag_service()
    events = service.answer_stream(
        question=request.question,
        top_k=request.top_k or settings.retrieval_top_k,
    )
    return EventSourceResponse(events)
```

- [ ] **Step 6: Run streaming tests and API import check**

Run:

```bash
.venv/bin/pytest tests/test_streaming.py -v
.venv/bin/python -c "from app.main import app; print([route.path for route in app.routes if 'ask' in route.path])"
```

Expected: tests PASS and route list includes `/ask` and `/ask/stream`.

- [ ] **Step 7: Commit**

Run:

```bash
git add app/rag/llm.py app/rag/service.py app/api/routes.py tests/test_streaming.py
git commit -m "feat: add streaming rag answers"
```

## Task 9: Lightweight Evaluation Dataset And Script

**Files:**
- Create: `app/evaluation/__init__.py`
- Create: `app/evaluation/dataset.py`
- Create: `app/evaluation/metrics.py`
- Create: `scripts/evaluate.py`
- Create: `data/eval/sample_eval.jsonl`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/test_evaluation.py`:

```python
import json
from pathlib import Path

from app.evaluation.dataset import load_eval_cases
from app.evaluation.metrics import hit_rate_at_k
from app.rag.documents import RetrievedDocument


def test_load_eval_cases_from_jsonl(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "RAG 是什么？",
                "ground_truth": "检索增强生成",
                "expected_sources": ["rag.md"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_eval_cases(path)

    assert cases[0].question == "RAG 是什么？"
    assert cases[0].expected_sources == ["rag.md"]


def test_hit_rate_at_k_counts_expected_source_match():
    retrieved = [
        [RetrievedDocument(id="1", content="x", source="rag.md", metadata={})],
        [RetrievedDocument(id="2", content="y", source="other.md", metadata={})],
    ]
    expected_sources = [["rag.md"], ["missing.md"]]

    assert hit_rate_at_k(retrieved, expected_sources) == 0.5
```

- [ ] **Step 2: Run evaluation tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -v
```

Expected: FAIL because `app.evaluation` does not exist.

- [ ] **Step 3: Implement evaluation dataset loader**

Create `app/evaluation/__init__.py` as an empty file.

Create `app/evaluation/dataset.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    question: str
    ground_truth: str
    expected_sources: list[str]


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            EvalCase(
                question=str(row["question"]),
                ground_truth=str(row.get("ground_truth", "")),
                expected_sources=[str(source) for source in row.get("expected_sources", [])],
            )
        )
    return cases
```

- [ ] **Step 4: Implement lightweight retrieval metric**

Create `app/evaluation/metrics.py`:

```python
from app.rag.documents import RetrievedDocument


def hit_rate_at_k(retrieved: list[list[RetrievedDocument]], expected_sources: list[list[str]]) -> float:
    if not expected_sources:
        return 0.0

    hits = 0
    for documents, expected in zip(retrieved, expected_sources, strict=False):
        expected_set = set(expected)
        retrieved_sources = {document.source for document in documents}
        if retrieved_sources & expected_set:
            hits += 1
    return hits / len(expected_sources)
```

- [ ] **Step 5: Add sample evaluation set**

Create `data/eval/sample_eval.jsonl`:

```jsonl
{"question":"RAG 系统包含哪些核心步骤？","ground_truth":"文档解析、分块、向量化、检索、提示词组装和大模型回答生成。","expected_sources":["data/documents/example.md"]}
{"question":"系统如何保证答案可追溯？","ground_truth":"接口会返回引用来源，包括文件路径、页码和 chunk 编号。","expected_sources":["data/documents/example.md"]}
```

- [ ] **Step 6: Implement evaluation script**

Create `scripts/evaluate.py`:

```python
from pathlib import Path

from app.api.routes import build_retriever
from app.core.config import get_settings
from app.evaluation.dataset import EvalCase
from app.evaluation.dataset import load_eval_cases
from app.evaluation.metrics import hit_rate_at_k
from app.rag.documents import chunk_to_retrieved_document
from app.rag.documents import RetrievedDocument


def retrieve_cases(cases: list[EvalCase], retriever, top_k: int) -> list[list[RetrievedDocument]]:
    retrieved: list[list[RetrievedDocument]] = []
    for case in cases:
        chunks = retriever.similarity_search(case.question, top_k=top_k)
        retrieved.append([chunk_to_retrieved_document(chunk) for chunk in chunks])
    return retrieved


def main() -> None:
    settings = get_settings()
    eval_path = Path("data/eval/sample_eval.jsonl")
    cases = load_eval_cases(eval_path)
    retriever = build_retriever()
    all_retrieved = retrieve_cases(cases, retriever, top_k=settings.retrieval_top_k)

    expected_sources = [case.expected_sources for case in cases]
    print({"cases": len(cases), "hit_rate_at_k": hit_rate_at_k(all_retrieved, expected_sources)})


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run evaluation tests**

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 8: Run evaluation script after ingestion**

Run:

```bash
.venv/bin/python -m scripts.ingest
.venv/bin/python -m scripts.evaluate
```

Expected: prints a dictionary containing `cases` and `hit_rate_at_k`. The exact score depends on local embedding behavior and indexed corpus.

- [ ] **Step 9: Commit**

Run:

```bash
git add app/evaluation scripts/evaluate.py data/eval/sample_eval.jsonl tests/test_evaluation.py
git commit -m "feat: add retrieval evaluation script"
```

## Task 10: README And Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with Phase 2 features**

Modify `README.md` to add a `第二阶段能力` section:

```markdown
## 第二阶段能力

- 支持 `docx`、`html`、`htm` 文档解析。
- 建立索引时会同时生成 Chroma 向量索引和 BM25 JSONL 语料。
- 问答默认走混合检索：Chroma 稠密检索 + BM25 稀疏检索 + RRF 融合。
- 混合检索候选结果会经过 BGE Reranker 精排。
- `/ask/stream` 支持 SSE 流式输出。
- `python -m scripts.evaluate` 可运行轻量级检索评估。
- `python -m scripts.warmup` 可预加载 embedding 和 reranker 模型，并实际触发一次检索和 rerank。
```

Add streaming example:

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG 系统包含哪些核心步骤？","top_k":4}'
```

Add evaluation example:

```bash
python -m scripts.ingest
python -m scripts.evaluate
```

- [ ] **Step 2: Run all tests**

Run:

```bash
.venv/bin/pytest -v
```

Expected: PASS.

- [ ] **Step 3: Verify app imports**

Run:

```bash
.venv/bin/python -c "from app.main import app; print(app.title)"
```

Expected:

```text
RAG Knowledge QA System
```

- [ ] **Step 4: Verify ingestion**

Run:

```bash
.venv/bin/python -m scripts.ingest
```

Expected: prints `IngestResult(...)` with `indexed_chunks` greater than `0` for the sample document.

- [ ] **Step 5: Verify evaluation script**

Run:

```bash
.venv/bin/python -m scripts.evaluate
```

Expected: prints a dictionary containing `cases` and `hit_rate_at_k`.

- [ ] **Step 6: Check git ignore hygiene**

Run:

```bash
git status --ignored --short | sed -n '1,160p'
```

Expected: `.env`, `.venv/`, `.pytest_cache/`, `__pycache__/`, and `data/chroma/` generated files are ignored. Source files and tests should not appear as ignored.

- [ ] **Step 7: Commit README**

Run:

```bash
git add README.md
git commit -m "docs: document phase 2 rag features"
```

- [ ] **Step 8: Push**

Run:

```bash
git push
```

Expected: remote `main` receives all Phase 2 commits.

## Plan Self-Review

- Spec coverage: This plan covers the main Phase 2 gap against the reference RAG project: richer document parsing, hybrid retrieval, RRF, reranking hook, streaming output, and evaluation scaffolding.
- Scope control: Milvus, Redis, K8s, image OCR, and production RAGAS metrics are intentionally left for Phase 3 because they add infrastructure and data-labeling work.
- Placeholder scan: No implementation steps rely on unspecified code. Each new module includes concrete code and targeted tests.
- Type consistency: `Chunk`, `RetrievedDocument`, `BM25Retriever`, `HybridRetriever`, `RagService`, and config field names are consistent across tasks.
- Verification path: The plan ends with full pytest, app import, ingestion, evaluation, git ignore hygiene, commit, and push.
