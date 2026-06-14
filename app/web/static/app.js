import { clampTopK, escapeHtml, formatPercent, normalizeSources, parseSseChunk } from "./ui-utils.js";

const elements = {
  healthStatus: document.querySelector("#healthStatus"),
  ingestButton: document.querySelector("#ingestButton"),
  ingestState: document.querySelector("#ingestState"),
  ingestLog: document.querySelector("#ingestLog"),
  evaluationButton: document.querySelector("#evaluationButton"),
  evaluationState: document.querySelector("#evaluationState"),
  evaluationMetrics: document.querySelector("#evaluationMetrics"),
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

async function readJsonResponse(response) {
  const contentType = response.headers.get("content-type") ?? "";

  if (contentType.includes("application/json")) {
    return response.json();
  }

  return { detail: await response.text() };
}

async function checkHealth() {
  try {
    const response = await fetch("/health");

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    setStatus(elements.healthStatus, "在线", "is-ok");
  } catch {
    setStatus(elements.healthStatus, "离线", "is-error");
  }
}

async function ingestDocuments() {
  elements.ingestButton.disabled = true;
  setStatus(elements.ingestState, "运行中");
  elements.ingestLog.textContent = "正在重建索引...";

  try {
    const response = await fetch("/ingest", { method: "POST" });
    const payload = await readJsonResponse(response);

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

function handleSseFrame(frame) {
  if (frame.event === "token") {
    elements.answerOutput.textContent += frame.data;
    return;
  }

  if (frame.event === "sources") {
    renderSources(JSON.parse(frame.data));
  }
}

function splitCompleteSseFrames(buffer) {
  const normalized = buffer.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
  const boundary = normalized.lastIndexOf("\n\n");

  if (boundary === -1) {
    return { complete: "", remainder: normalized };
  }

  return {
    complete: normalized.slice(0, boundary + 2),
    remainder: normalized.slice(boundary + 2),
  };
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
      const { complete, remainder } = splitCompleteSseFrames(buffer);

      if (!complete) {
        buffer = remainder;
        continue;
      }

      buffer = remainder;
      parseSseChunk(complete).forEach(handleSseFrame);
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      parseSseChunk(buffer).forEach(handleSseFrame);
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
elements.evaluationButton.addEventListener("click", runEvaluation);

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
