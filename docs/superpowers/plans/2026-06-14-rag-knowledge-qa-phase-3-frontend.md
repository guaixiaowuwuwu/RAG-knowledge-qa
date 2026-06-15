# RAG Knowledge QA Phase 3 Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a polished local web UI for the RAG knowledge QA system so users can ingest documents, ask questions, stream answers, inspect citations, and see retrieval/evaluation status from a browser.

**Architecture:** Keep the backend as the single FastAPI app and serve a static frontend from `app/web/static`. The frontend uses plain HTML/CSS/JavaScript, talks to existing `/health`, `/ingest`, `/ask`, and `/ask/stream` APIs, and includes small pure functions for SSE parsing and citation rendering that can be unit-tested without a browser.

**Tech Stack:** FastAPI static files, vanilla HTML/CSS/JavaScript, pytest for backend/static route tests, Node.js built-in test runner for frontend utility tests, Playwright/browser verification for final visual and interaction checks.

---

## Scope

### In Scope

- Serve a browser UI at `/`.
- Keep existing API routes unchanged.
- Add a focused dashboard-style interface for:
  - API health status.
  - Document ingestion trigger.
  - Warmup guidance.
  - Question input with `top_k` control.
  - Streaming answer display.
  - Citation/source list.
  - Lightweight evaluation trigger instructions.
- Add frontend utilities for SSE event parsing, citation normalization, and UI-safe escaping.
- Add backend tests that verify static assets are mounted.
- Add JavaScript tests for frontend utilities.
- Update README with frontend usage.

### Out of Scope

- File upload from the browser.
- Authentication.
- Multi-user sessions.
- Chat history persistence.
- React/Vite/Next.js build pipeline.
- Production deployment and CDN.

The UI should make the current system pleasant to demo. It should not turn this Python RAG project into a frontend framework festival. One circus at a time.

## Design Direction

This is an operational AI tool, not a marketing landing page. The first screen should be the actual work surface:

- Left column: system status, ingestion, warmup/evaluation notes.
- Main column: question composer, streaming answer, source citations.
- Right or lower panel on smaller screens: retrieval settings and recent request metadata.

Visual style:

- Quiet, technical, readable.
- Dense but not cramped.
- Strong contrast, clear source cards, no oversized hero.
- Cards only for actual panels and repeated citation items.
- Stable dimensions for the answer area and controls to avoid layout jumping during streaming.

## File Structure

- Modify: `pyproject.toml` if `httpx` is needed for FastAPI `TestClient`.
- Modify: `app/main.py` to mount static assets and root page.
- Create: `app/web/__init__.py`.
- Create: `app/web/static/index.html`.
- Create: `app/web/static/styles.css`.
- Create: `app/web/static/app.js`.
- Create: `app/web/static/ui-utils.js`.
- Create: `tests/test_web_static.py`.
- Create: `tests/js/ui-utils.test.mjs`.
- Modify: `README.md`.
- Modify: `.gitignore` only if frontend generated files are introduced.

## Task 1: Static Web Mount

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/main.py`
- Create: `app/web/__init__.py`
- Create: `app/web/static/index.html`
- Create: `app/web/static/styles.css`
- Create: `app/web/static/app.js`
- Create: `tests/test_web_static.py`

- [ ] **Step 1: Write failing backend static route tests**

Create `tests/test_web_static.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_root_serves_frontend_html():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "RAG Knowledge QA" in response.text
    assert "/static/styles.css" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_are_served():
    client = TestClient(app)

    css_response = client.get("/static/styles.css")
    js_response = client.get("/static/app.js")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
```

- [ ] **Step 2: Run static route tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_web_static.py -v
```

Expected: FAIL because `/` and `/static/*` are not mounted.

- [ ] **Step 3: Ensure TestClient dependency is available**

Check whether `httpx` is already installed:

```bash
.venv/bin/python -c "import httpx; print(httpx.__version__)"
```

If it fails, add this dependency to `pyproject.toml`:

```toml
    "httpx>=0.27.0",
```

Then run:

```bash
.venv/bin/pip install -e ".[dev]"
```

Expected: installation completes without dependency errors.

- [ ] **Step 4: Create frontend package marker**

Create `app/web/__init__.py` as an empty file.

- [ ] **Step 5: Create placeholder HTML**

Create `app/web/static/index.html`:

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RAG Knowledge QA</title>
    <link rel="stylesheet" href="/static/styles.css" />
  </head>
  <body>
    <main class="app-shell">
      <section class="panel">
        <p class="eyebrow">RAG Knowledge QA</p>
        <h1>企业知识库问答</h1>
        <p>文档索引、混合检索、Reranker 精排和引用溯源。</p>
      </section>
    </main>
    <script type="module" src="/static/app.js"></script>
  </body>
</html>
```

- [ ] **Step 6: Create placeholder CSS**

Create `app/web/static/styles.css`:

```css
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f4f6f8;
  color: #17202a;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  background: #f4f6f8;
}

.app-shell {
  width: min(1180px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 28px 0;
}

.panel {
  background: #ffffff;
  border: 1px solid #dbe2ea;
  border-radius: 8px;
  padding: 20px;
}

.eyebrow {
  margin: 0 0 8px;
  color: #2f6fed;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0;
}
```

- [ ] **Step 7: Create placeholder app script**

Create `app/web/static/app.js`:

```javascript
console.info("RAG Knowledge QA frontend loaded");
```

- [ ] **Step 8: Mount static frontend in FastAPI**

Modify `app/main.py`:

```python
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router


WEB_DIR = Path(__file__).parent / "web" / "static"

app = FastAPI(title="RAG Knowledge QA System")
app.include_router(router)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")
```

- [ ] **Step 9: Run static route tests and verify pass**

Run:

```bash
.venv/bin/pytest tests/test_web_static.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

Run:

```bash
git add pyproject.toml app/main.py app/web tests/test_web_static.py
git commit -m "feat: serve frontend shell"
```

## Task 2: Frontend Utility Functions

**Files:**
- Create: `app/web/static/ui-utils.js`
- Create: `tests/js/ui-utils.test.mjs`

- [ ] **Step 1: Write failing JavaScript utility tests**

Create `tests/js/ui-utils.test.mjs`:

```javascript
import assert from "node:assert/strict";
import test from "node:test";

import {
  escapeHtml,
  parseSseChunk,
  normalizeSources,
  clampTopK,
} from "../../app/web/static/ui-utils.js";

test("escapeHtml escapes dangerous characters", () => {
  assert.equal(
    escapeHtml('<script>alert("x")</script>'),
    "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;",
  );
});

test("parseSseChunk parses event source frames", () => {
  const frames = parseSseChunk('event: token\ndata: 你好\n\nevent: sources\ndata: []\n\n');

  assert.deepEqual(frames, [
    { event: "token", data: "你好" },
    { event: "sources", data: "[]" },
  ]);
});

test("normalizeSources fills missing optional fields", () => {
  const sources = normalizeSources([
    { source: "data/documents/example.md", content: "RAG 内容" },
  ]);

  assert.deepEqual(sources, [
    {
      source: "data/documents/example.md",
      page: null,
      chunk_index: null,
      content: "RAG 内容",
    },
  ]);
});

test("clampTopK keeps values within API bounds", () => {
  assert.equal(clampTopK("0"), 1);
  assert.equal(clampTopK("25"), 20);
  assert.equal(clampTopK("4"), 4);
});
```

- [ ] **Step 2: Run JS tests and verify failure**

Run:

```bash
node --test tests/js/ui-utils.test.mjs
```

Expected: FAIL because `ui-utils.js` does not export these functions.

- [ ] **Step 3: Implement frontend utilities**

Create `app/web/static/ui-utils.js`:

```javascript
export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function parseSseChunk(chunkText) {
  return chunkText
    .split("\n\n")
    .map((frame) => frame.trim())
    .filter(Boolean)
    .map((frame) => {
      const lines = frame.split("\n");
      let event = "message";
      const dataLines = [];
      for (const line of lines) {
        if (line.startsWith("event:")) {
          event = line.slice("event:".length).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice("data:".length).trimStart());
        }
      }
      return { event, data: dataLines.join("\n") };
    });
}

export function normalizeSources(sources) {
  if (!Array.isArray(sources)) {
    return [];
  }
  return sources.map((source) => ({
    source: source.source ?? "",
    page: source.page ?? null,
    chunk_index: source.chunk_index ?? null,
    content: source.content ?? "",
  }));
}

export function clampTopK(value) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) {
    return 4;
  }
  return Math.min(20, Math.max(1, parsed));
}
```

- [ ] **Step 4: Run JS tests and verify pass**

Run:

```bash
node --test tests/js/ui-utils.test.mjs
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add app/web/static/ui-utils.js tests/js/ui-utils.test.mjs
git commit -m "feat: add frontend utility tests"
```

## Task 3: Full Frontend Layout And Styling

**Files:**
- Modify: `app/web/static/index.html`
- Modify: `app/web/static/styles.css`

- [ ] **Step 1: Replace HTML shell with full work surface**

Modify `app/web/static/index.html`:

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RAG Knowledge QA</title>
    <link rel="stylesheet" href="/static/styles.css" />
  </head>
  <body>
    <main class="app-shell">
      <header class="topbar">
        <div>
          <p class="eyebrow">RAG Knowledge QA</p>
          <h1>企业知识库问答控制台</h1>
        </div>
        <div class="status-pill" id="healthStatus">检查中</div>
      </header>

      <section class="workspace">
        <aside class="side-panel" aria-label="系统操作">
          <section class="panel">
            <div class="panel-heading">
              <h2>索引</h2>
              <span id="ingestState">未运行</span>
            </div>
            <p class="muted">扫描 `data/documents`，重建 Chroma 向量索引和 BM25 语料。</p>
            <button class="primary-button" id="ingestButton" type="button">重建索引</button>
            <pre class="mini-log" id="ingestLog">等待操作</pre>
          </section>

          <section class="panel">
            <div class="panel-heading">
              <h2>检索链路</h2>
            </div>
            <ol class="pipeline-list">
              <li>Chroma dense retrieval</li>
              <li>BM25 sparse retrieval</li>
              <li>RRF fusion</li>
              <li>BGE reranker</li>
              <li>LLM answer with citations</li>
            </ol>
          </section>

          <section class="panel">
            <div class="panel-heading">
              <h2>演示前</h2>
            </div>
            <p class="muted">首次加载本地 embedding 和 reranker 会较慢，演示前先在终端运行：</p>
            <code class="command">python -m scripts.warmup</code>
          </section>
        </aside>

        <section class="main-panel">
          <form class="ask-form" id="askForm">
            <label for="questionInput">问题</label>
            <textarea id="questionInput" rows="4" placeholder="例如：RAG 系统包含哪些核心步骤？"></textarea>

            <div class="form-row">
              <label class="compact-label" for="topKInput">Top K</label>
              <input id="topKInput" type="number" min="1" max="20" value="4" />
              <button class="primary-button" id="askButton" type="submit">流式提问</button>
              <button class="secondary-button" id="clearButton" type="button">清空</button>
            </div>
          </form>

          <section class="answer-panel" aria-live="polite">
            <div class="panel-heading">
              <h2>回答</h2>
              <span id="answerState">等待问题</span>
            </div>
            <div class="answer-output" id="answerOutput">答案会实时显示在这里。</div>
          </section>

          <section class="sources-panel">
            <div class="panel-heading">
              <h2>引用来源</h2>
              <span id="sourceCount">0 条</span>
            </div>
            <div class="sources-list" id="sourcesList">
              <p class="muted">完成一次问答后显示检索片段、文件路径、页码和 chunk 编号。</p>
            </div>
          </section>
        </section>
      </section>
    </main>

    <script type="module" src="/static/app.js"></script>
  </body>
</html>
```

- [ ] **Step 2: Replace CSS with responsive dashboard styling**

Modify `app/web/static/styles.css`:

```css
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f3f5f7;
  color: #17202a;
  --surface: #ffffff;
  --surface-soft: #f8fafc;
  --line: #d9e1ea;
  --line-strong: #b8c4d2;
  --text-muted: #64748b;
  --accent: #2563eb;
  --accent-strong: #1d4ed8;
  --danger: #b42318;
  --success: #147a4d;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  background: var(--surface-soft);
}

button,
textarea,
input {
  font: inherit;
}

.app-shell {
  width: min(1280px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 24px 0 36px;
}

.topbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 18px;
}

.eyebrow {
  margin: 0 0 8px;
  color: var(--accent);
  font-size: 13px;
  font-weight: 750;
  letter-spacing: 0;
}

h1,
h2 {
  margin: 0;
  letter-spacing: 0;
}

h1 {
  font-size: clamp(28px, 4vw, 44px);
  line-height: 1.04;
}

h2 {
  font-size: 16px;
}

.status-pill,
.panel-heading span {
  min-width: 74px;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px 10px;
  background: var(--surface);
  color: var(--text-muted);
  font-size: 13px;
  text-align: center;
}

.workspace {
  display: grid;
  grid-template-columns: 330px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.side-panel,
.main-panel {
  display: grid;
  gap: 16px;
}

.panel,
.ask-form,
.answer-panel,
.sources-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  padding: 18px;
}

.panel-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.muted {
  color: var(--text-muted);
  line-height: 1.6;
}

.primary-button,
.secondary-button {
  min-height: 42px;
  border-radius: 7px;
  padding: 0 15px;
  border: 1px solid transparent;
  cursor: pointer;
  font-weight: 720;
}

.primary-button {
  background: var(--accent);
  color: #ffffff;
}

.primary-button:hover {
  background: var(--accent-strong);
}

.primary-button:disabled,
.secondary-button:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.secondary-button {
  border-color: var(--line-strong);
  background: #ffffff;
  color: #17202a;
}

.mini-log,
.command {
  display: block;
  width: 100%;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: #0f172a;
  color: #dbeafe;
  padding: 12px;
  font-size: 13px;
  line-height: 1.5;
}

.pipeline-list {
  margin: 0;
  padding-left: 20px;
  color: #334155;
  line-height: 1.8;
}

.ask-form {
  display: grid;
  gap: 12px;
}

.ask-form label {
  font-weight: 760;
}

textarea {
  width: 100%;
  resize: vertical;
  min-height: 120px;
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  padding: 14px;
  background: #ffffff;
  color: #17202a;
  line-height: 1.55;
}

.form-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.compact-label {
  color: var(--text-muted);
}

input[type="number"] {
  width: 88px;
  min-height: 42px;
  border: 1px solid var(--line-strong);
  border-radius: 7px;
  padding: 0 10px;
}

.answer-output {
  min-height: 260px;
  white-space: pre-wrap;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfdff;
  padding: 16px;
  line-height: 1.7;
}

.sources-list {
  display: grid;
  gap: 10px;
}

.source-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
  padding: 14px;
}

.source-meta {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 8px;
  color: var(--text-muted);
  font-size: 13px;
}

.source-content {
  margin: 0;
  color: #253244;
  line-height: 1.6;
}

.is-ok {
  color: var(--success);
  border-color: rgba(20, 122, 77, 0.25);
  background: #ecfdf3;
}

.is-error {
  color: var(--danger);
  border-color: rgba(180, 35, 24, 0.25);
  background: #fff1f0;
}

@media (max-width: 880px) {
  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .workspace {
    grid-template-columns: 1fr;
  }

  .app-shell {
    width: min(100vw - 20px, 760px);
    padding-top: 16px;
  }
}
```

- [ ] **Step 3: Run static tests**

Run:

```bash
.venv/bin/pytest tests/test_web_static.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add app/web/static/index.html app/web/static/styles.css
git commit -m "feat: build rag frontend layout"
```

## Task 4: Browser App Behavior

**Files:**
- Modify: `app/web/static/app.js`
- Test: `tests/js/ui-utils.test.mjs`

- [ ] **Step 1: Implement browser behavior**

Modify `app/web/static/app.js`:

```javascript
import { clampTopK, escapeHtml, normalizeSources, parseSseChunk } from "./ui-utils.js";

const elements = {
  healthStatus: document.querySelector("#healthStatus"),
  ingestButton: document.querySelector("#ingestButton"),
  ingestState: document.querySelector("#ingestState"),
  ingestLog: document.querySelector("#ingestLog"),
  askForm: document.querySelector("#askForm"),
  questionInput: document.querySelector("#questionInput"),
  topKInput: document.querySelector("#topKInput"),
  askButton: document.querySelector("#askButton"),
  clearButton: document.querySelector("#clearButton"),
  answerState: document.querySelector("#answerState"),
  answerOutput: document.querySelector("#answerOutput"),
  sourceCount: document.querySelector("#sourceCount"),
  sourcesList: document.querySelector("#sourcesList"),
};

let currentController = null;

function setStatus(element, text, mode = "") {
  element.textContent = text;
  element.classList.remove("is-ok", "is-error");
  if (mode) {
    element.classList.add(mode);
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    setStatus(elements.healthStatus, "在线", "is-ok");
  } catch (error) {
    setStatus(elements.healthStatus, "离线", "is-error");
  }
}

async function ingestDocuments() {
  elements.ingestButton.disabled = true;
  setStatus(elements.ingestState, "运行中");
  elements.ingestLog.textContent = "正在重建索引...";

  try {
    const response = await fetch("/ingest", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(JSON.stringify(payload));
    }
    setStatus(elements.ingestState, "完成", "is-ok");
    elements.ingestLog.textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    setStatus(elements.ingestState, "失败", "is-error");
    elements.ingestLog.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    elements.ingestButton.disabled = false;
  }
}

function renderSources(sources) {
  const normalized = normalizeSources(sources);
  elements.sourceCount.textContent = `${normalized.length} 条`;

  if (normalized.length === 0) {
    elements.sourcesList.innerHTML = '<p class="muted">没有引用来源。</p>';
    return;
  }

  elements.sourcesList.innerHTML = normalized
    .map((source) => {
      const page = source.page === null ? "无页码" : `第 ${escapeHtml(source.page)} 页`;
      const chunk = source.chunk_index === null ? "无 chunk" : `chunk ${escapeHtml(source.chunk_index)}`;
      return `
        <article class="source-card">
          <div class="source-meta">
            <span>${escapeHtml(source.source)}</span>
            <span>${page}</span>
            <span>${chunk}</span>
          </div>
          <p class="source-content">${escapeHtml(source.content)}</p>
        </article>
      `;
    })
    .join("");
}

async function askStreaming(question, topK) {
  if (currentController) {
    currentController.abort();
  }

  currentController = new AbortController();
  elements.askButton.disabled = true;
  elements.answerOutput.textContent = "";
  renderSources([]);
  setStatus(elements.answerState, "生成中");

  try {
    const response = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK }),
      signal: currentController.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const boundary = buffer.lastIndexOf("\n\n");
      if (boundary === -1) {
        continue;
      }

      const complete = buffer.slice(0, boundary + 2);
      buffer = buffer.slice(boundary + 2);
      for (const frame of parseSseChunk(complete)) {
        if (frame.event === "token") {
          elements.answerOutput.textContent += frame.data;
        }
        if (frame.event === "sources") {
          renderSources(JSON.parse(frame.data));
        }
      }
    }

    setStatus(elements.answerState, "完成", "is-ok");
  } catch (error) {
    if (error.name !== "AbortError") {
      setStatus(elements.answerState, "失败", "is-error");
      elements.answerOutput.textContent = error instanceof Error ? error.message : String(error);
    }
  } finally {
    elements.askButton.disabled = false;
    currentController = null;
  }
}

elements.ingestButton.addEventListener("click", ingestDocuments);

elements.askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = elements.questionInput.value.trim();
  const topK = clampTopK(elements.topKInput.value);
  elements.topKInput.value = String(topK);

  if (!question) {
    setStatus(elements.answerState, "请输入问题", "is-error");
    return;
  }

  askStreaming(question, topK);
});

elements.clearButton.addEventListener("click", () => {
  if (currentController) {
    currentController.abort();
  }
  elements.questionInput.value = "";
  elements.answerOutput.textContent = "答案会实时显示在这里。";
  renderSources([]);
  setStatus(elements.answerState, "等待问题");
});

checkHealth();
```

- [ ] **Step 2: Run JavaScript tests**

Run:

```bash
node --test tests/js/ui-utils.test.mjs
```

Expected: PASS.

- [ ] **Step 3: Run backend tests**

Run:

```bash
.venv/bin/pytest tests/test_web_static.py tests/test_streaming.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add app/web/static/app.js
git commit -m "feat: connect frontend to rag api"
```

## Task 5: Frontend Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README frontend section**

Add this section to `README.md` before `## 调用接口`:

```markdown
## 前端页面

启动服务后打开：

```bash
uvicorn app.main:app --reload
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

页面支持健康检查、重建索引、流式提问和引用来源查看。演示前建议先运行：

```bash
python -m scripts.ingest
python -m scripts.warmup
```
```

Add frontend verification commands under the test section:

```markdown
```bash
node --test tests/js/ui-utils.test.mjs
```
```

- [ ] **Step 2: Run full Python tests**

Run:

```bash
.venv/bin/pytest -v
```

Expected: PASS.

- [ ] **Step 3: Run JavaScript tests**

Run:

```bash
node --test tests/js/ui-utils.test.mjs
```

Expected: PASS.

- [ ] **Step 4: Verify app imports and routes**

Run:

```bash
.venv/bin/python -c "from app.main import app; print(app.title); print(any(route.path == '/' for route in app.routes))"
```

Expected:

```text
RAG Knowledge QA System
True
```

- [ ] **Step 5: Start server for manual browser verification**

Run:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If port `8000` is occupied, use:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Expected: server starts without import errors.

- [ ] **Step 6: Browser verification**

Open the in-app browser or normal browser at:

```text
http://127.0.0.1:8000/
```

Verify:

- Page loads with the title `企业知识库问答控制台`.
- Health status becomes `在线`.
- `重建索引` shows JSON result with `indexed_chunks`.
- Asking `RAG 系统包含哪些核心步骤？` streams an answer.
- Citation list shows at least one source card.
- At mobile width around `390px`, controls do not overlap and source cards remain readable.

- [ ] **Step 7: Stop server**

Stop the `uvicorn` process with `Ctrl+C`.

- [ ] **Step 8: Commit docs**

Run:

```bash
git add README.md
git commit -m "docs: document frontend workflow"
```

- [ ] **Step 9: Push**

Run:

```bash
git push
```

Expected: remote `main` receives all Phase 3 frontend commits.

## Plan Self-Review

- Spec coverage: The plan adds a browser UI, static serving, ingest controls, streaming answer display, citations, tests, README usage, and manual browser verification.
- Scope control: File upload, auth, chat persistence, React, deployment, and RAGAS remain out of this phase.
- Placeholder scan: No implementation step relies on missing code or unspecified behavior.
- Type consistency: Frontend JS uses existing `AskRequest`, `/ingest`, `/ask/stream`, and source response fields exactly as defined in the backend.
- Verification path: The plan ends with Python tests, Node tests, route import check, server startup, browser verification, commit, and push.
