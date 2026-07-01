import {
  basename,
  clampTopK,
  escapeHtml,
  formatBytes,
  formatDateTime,
  formatInteger,
  formatPercent,
  hitLabel,
  groupedMetricRows,
  normalizeSources,
  parseSseChunk,
} from "./ui-utils.js?v=phase10";

const elements = {
  healthStatus: document.querySelector("#healthStatus"),
  corpusState: document.querySelector("#corpusState"),
  corpusStatusGrid: document.querySelector("#corpusStatusGrid"),
  corpusStatusMeta: document.querySelector("#corpusStatusMeta"),
  ingestButton: document.querySelector("#ingestButton"),
  ingestState: document.querySelector("#ingestState"),
  ingestLog: document.querySelector("#ingestLog"),
  evaluationButton: document.querySelector("#evaluationButton"),
  evaluationState: document.querySelector("#evaluationState"),
  evaluationMetrics: document.querySelector("#evaluationMetrics"),
  evaluationReportMeta: document.querySelector("#evaluationReportMeta"),
  evaluationGroups: document.querySelector("#evaluationGroups"),
  ragasButton: document.querySelector("#ragasButton"),
  ragasState: document.querySelector("#ragasState"),
  ragasMetrics: document.querySelector("#ragasMetrics"),
  ragasReportMeta: document.querySelector("#ragasReportMeta"),
  presetQuestions: document.querySelector("#presetQuestions"),
  askForm: document.querySelector("#askForm"),
  questionInput: document.querySelector("#questionInput"),
  topKInput: document.querySelector("#topKInput"),
  rewriteToggle: document.querySelector("#rewriteToggle"),
  hydeToggle: document.querySelector("#hydeToggle"),
  parentToggle: document.querySelector("#parentToggle"),
  askButton: document.querySelector("#askButton"),
  clearButton: document.querySelector("#clearButton"),
  answerState: document.querySelector("#answerState"),
  answerOutput: document.querySelector("#answerOutput"),
  feedbackGoodButton: document.querySelector("#feedbackGoodButton"),
  feedbackBadButton: document.querySelector("#feedbackBadButton"),
  feedbackState: document.querySelector("#feedbackState"),
  sourceCount: document.querySelector("#sourceCount"),
  sourcesList: document.querySelector("#sourcesList"),
  reportState: document.querySelector("#reportState"),
  comparisonReport: document.querySelector("#comparisonReport"),
  evaluationCases: document.querySelector("#evaluationCases"),
  debugState: document.querySelector("#debugState"),
  debugContent: document.querySelector("#debugContent"),
};

const presets = [
  {
    label: "BYD 年报",
    question: "比亚迪 2024 年年度报告披露的营业额是多少？",
  },
  {
    label: "BYD 制度",
    question: "比亚迪信息披露事务管理制度要求信息披露遵循哪些基本原则？",
  },
  {
    label: "SEC 10-K",
    question: "What reportable segments does Apple list in its 2025 Form 10-K?",
  },
  {
    label: "负例",
    question: "比亚迪今天的股票收盘价是多少？",
  },
];

let currentController = null;
let streamHasToken = false;
let currentSessionId = null;

function setStatus(element, text, mode = "") {
  element.textContent = text;
  element.classList.remove("is-ok", "is-error", "is-warn");

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

function renderPresets() {
  elements.presetQuestions.innerHTML = presets
    .map(
      (preset) => `
        <button class="preset-button" type="button" data-question="${escapeHtml(preset.question)}">
          ${escapeHtml(preset.label)}
        </button>
      `,
    )
    .join("");
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

function statusCard(value, label, detail = "") {
  return `
    <div>
      <strong>${escapeHtml(value)}</strong>
      <span>${escapeHtml(label)}</span>
      ${detail ? `<small>${escapeHtml(detail)}</small>` : ""}
    </div>
  `;
}

function renderCorpusStatus(payload) {
  elements.corpusStatusGrid.innerHTML = [
    statusCard(payload.ready ? "ready" : "pending", "Index state", payload.readiness_reason ?? ""),
    statusCard(formatInteger(payload.document_count), "Documents"),
    statusCard(formatInteger(payload.chunk_count), "Chunks"),
    statusCard(formatInteger(payload.parent_chunk_count), "Parent chunks"),
    statusCard(payload.bm25_ready ? "ready" : "missing", "BM25 corpus"),
    statusCard(
      formatInteger(payload.chroma?.chunk_count),
      "Chroma rows",
      payload.chroma_collection_name ?? payload.chroma?.collection_name ?? "",
    ),
    statusCard(formatBytes(payload.chroma?.size_bytes), "Index size"),
  ].join("");

  const meta = [
    `文档目录：${payload.documents_dir ?? ""}`,
    `BM25：${payload.bm25_corpus?.path ?? ""}`,
    `Chroma：${payload.chroma?.persist_dir ?? ""}`,
    `更新时间：${formatDateTime(payload.chroma?.updated_at ?? payload.bm25_corpus?.updated_at)}`,
  ];

  if (payload.chroma?.error) {
    meta.push(`Chroma 状态错误：${payload.chroma.error}`);
  }

  elements.corpusStatusMeta.innerHTML = meta.map((item) => `<span>${escapeHtml(item)}</span>`).join("");

  if (payload.ready) {
    setStatus(elements.corpusState, "已就绪", "is-ok");
  } else if (payload.document_count > 0) {
    setStatus(elements.corpusState, "待索引", "is-warn");
  } else {
    setStatus(elements.corpusState, "空语料", "is-error");
  }
}

async function loadCorpusStatus() {
  setStatus(elements.corpusState, "加载中");
  elements.corpusStatusGrid.innerHTML = '<span class="state-placeholder">正在读取本地索引状态</span>';
  elements.corpusStatusMeta.textContent = "";

  try {
    const response = await fetch("/corpus/status");
    const payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(JSON.stringify(payload));
    }

    renderCorpusStatus(payload);
  } catch (error) {
    setStatus(elements.corpusState, "失败", "is-error");
    elements.corpusStatusGrid.innerHTML = `<span class="state-placeholder">${escapeHtml(
      error instanceof Error ? error.message : String(error),
    )}</span>`;
  }
}

async function ingestDocuments() {
  elements.ingestButton.disabled = true;
  setStatus(elements.ingestState, "运行中");
  setStatus(elements.corpusState, "重建中");
  elements.ingestLog.textContent = "正在重建索引...";

  try {
    const response = await fetch("/ingest", { method: "POST" });
    const payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(JSON.stringify(payload));
    }

    setStatus(elements.ingestState, "完成", "is-ok");
    elements.ingestLog.textContent = JSON.stringify(payload, null, 2);
    await loadCorpusStatus();
  } catch (error) {
    setStatus(elements.ingestState, "失败", "is-error");
    elements.ingestLog.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    elements.ingestButton.disabled = false;
  }
}

function renderEvaluationMetrics(summary) {
  elements.evaluationMetrics.innerHTML = `
    ${statusCard(formatInteger(summary.cases), "Cases", `${formatInteger(summary.positive_cases)} positive / ${formatInteger(summary.negative_cases)} negative`)}
    ${statusCard(formatPercent(summary.hit_rate_at_k), "Hit@K")}
    ${statusCard(formatPercent(summary.mrr_at_k), "MRR@K")}
    ${statusCard(formatPercent(summary.source_recall_at_k ?? summary.source_recall), "Source Recall")}
    ${statusCard(formatPercent(summary.precision_at_k), "Precision@K")}
    ${statusCard(formatPercent(summary.ndcg_at_k), "NDCG@K")}
    ${statusCard(formatPercent(summary.negative_rejection_rate), "Negative Reject")}
  `;
}

function renderGroupedMetrics(groups) {
  const rows = groupedMetricRows(groups);

  if (rows.length === 0) {
    elements.evaluationGroups.innerHTML = '<p class="muted">当前报告没有分组指标。</p>';
    return;
  }

  elements.evaluationGroups.innerHTML = `
    <table class="grouped-metric-table">
      <thead>
        <tr>
          <th>Dimension</th>
          <th>Group</th>
          <th>Cases</th>
          <th>Hit@K</th>
          <th>MRR</th>
          <th>Neg Reject</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map((row) => {
            const summary = row.summary ?? {};
            return `
              <tr>
                <td>${escapeHtml(row.dimension)}</td>
                <td>${escapeHtml(row.label)}</td>
                <td>${escapeHtml(formatInteger(summary.cases))}</td>
                <td>${escapeHtml(formatPercent(summary.hit_rate_at_k))}</td>
                <td>${escapeHtml(formatPercent(summary.mrr_at_k))}</td>
                <td>${escapeHtml(formatPercent(summary.negative_rejection_rate))}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderEvaluationMeta(payload) {
  const parts = [
    payload.variant ? `变体：${payload.variant}` : "",
    payload.dataset_path ? `数据集：${payload.dataset_path}` : "",
    payload.top_k ? `Top K：${payload.top_k}` : "",
    payload.generated_at ? `生成：${formatDateTime(payload.generated_at)}` : "",
  ].filter(Boolean);

  elements.evaluationReportMeta.innerHTML = parts.map((part) => `<span>${escapeHtml(part)}</span>`).join("");
}

function renderEvaluationCases(cases) {
  if (!Array.isArray(cases) || cases.length === 0) {
    elements.evaluationCases.innerHTML = '<p class="muted">当前报告没有样本明细。</p>';
    return;
  }

  const misses = cases.filter((item) => !item.hit && !item.is_negative);
  const negativeErrors = cases.filter((item) => item.hit && item.is_negative);
  const preview = [...misses, ...negativeErrors, ...cases].slice(0, 14);

  elements.evaluationCases.innerHTML = preview
    .map((item) => {
      const label = hitLabel(item.hit, item.is_negative);
      const mode = item.is_negative ? (item.hit ? "is-error" : "is-ok") : item.hit ? "is-ok" : "is-error";
      const expected = Array.isArray(item.expected_sources) ? item.expected_sources.map(basename).join(", ") : "无";
      const retrieved = Array.isArray(item.retrieved_sources)
        ? item.retrieved_sources.slice(0, 4).map(basename).join(", ")
        : "无";

      return `
        <article class="case-card">
          <div class="case-card-header">
            <strong>${escapeHtml(item.id || "case")}</strong>
            <span class="${mode}">${escapeHtml(label)}</span>
          </div>
          <p>${escapeHtml(item.question ?? "")}</p>
          <dl>
            <div><dt>Expected</dt><dd>${escapeHtml(expected || "无")}</dd></div>
            <div><dt>Retrieved</dt><dd>${escapeHtml(retrieved || "无")}</dd></div>
          </dl>
        </article>
      `;
    })
    .join("");
}

async function runEvaluation() {
  elements.evaluationButton.disabled = true;
  setStatus(elements.evaluationState, "运行中");
  setStatus(elements.reportState, "运行中");
  elements.evaluationMetrics.innerHTML = '<span class="state-placeholder">正在计算检索指标</span>';
  elements.evaluationCases.innerHTML = '<p class="muted">正在等待样本明细。</p>';

  try {
    const response = await fetch("/evaluation/report");
    const payload = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(JSON.stringify(payload));
    }

    renderEvaluationMetrics(payload.summary ?? {});
    renderGroupedMetrics(payload.groups ?? {});
    renderEvaluationMeta(payload);
    renderEvaluationCases(payload.cases);
    setStatus(elements.evaluationState, "完成", "is-ok");
    setStatus(elements.reportState, "已更新", "is-ok");
    await loadComparisonReport();
  } catch (error) {
    setStatus(elements.evaluationState, "失败", "is-error");
    setStatus(elements.reportState, "失败", "is-error");
    elements.evaluationMetrics.innerHTML = `<span class="state-placeholder">${escapeHtml(
      error instanceof Error ? error.message : String(error),
    )}</span>`;
  } finally {
    elements.evaluationButton.disabled = false;
  }
}

function renderComparisonReport(payload) {
  if (!payload.available) {
    setStatus(elements.reportState, "无对比", "is-warn");
    elements.comparisonReport.innerHTML = `<p class="muted">${escapeHtml(payload.error ?? "未找到对比报告。")}</p>`;
    return;
  }

  const rows = Array.isArray(payload.variants) ? payload.variants : [];
  if (rows.length === 0) {
    setStatus(elements.reportState, "无对比", "is-warn");
    elements.comparisonReport.innerHTML = '<p class="muted">最新对比报告没有变体指标。</p>';
    return;
  }

  setStatus(elements.reportState, "有对比", "is-ok");
  elements.comparisonReport.innerHTML = `
    <table class="comparison-table">
      <thead>
        <tr>
          <th>Variant</th>
          <th>Cases</th>
          <th>Hit@K</th>
          <th>MRR</th>
          <th>Recall</th>
          <th>NDCG</th>
          <th>Neg Reject</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map((row) => {
            const summary = row.summary ?? {};
            return `
              <tr>
                <td>${escapeHtml(row.variant ?? "unknown")}</td>
                <td>${escapeHtml(formatInteger(summary.cases))}</td>
                <td>${escapeHtml(formatPercent(summary.hit_rate_at_k))}</td>
                <td>${escapeHtml(formatPercent(summary.mrr_at_k))}</td>
                <td>${escapeHtml(formatPercent(summary.source_recall_at_k ?? summary.source_recall))}</td>
                <td>${escapeHtml(formatPercent(summary.ndcg_at_k))}</td>
                <td>${escapeHtml(formatPercent(summary.negative_rejection_rate))}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
    <div class="comparison-groups">
      ${rows
        .map((row) => {
          const groups = row.groups ?? {};
          const groupedRows = groupedMetricRows(groups, ["language", "category", "difficulty", "is_negative"]);
          const preview = groupedRows
            .slice(0, 4)
            .map((groupRow) => `${groupRow.dimension}:${groupRow.label}`)
            .join(" · ");
          return `
            <div class="comparison-group-preview">
              <strong>${escapeHtml(row.variant ?? "unknown")}</strong>
              <span>${escapeHtml(preview || "no groups")}</span>
            </div>
          `;
        })
        .join("")}
    </div>
    <div class="report-meta compact">
      <span>${escapeHtml(payload.report_path ?? "")}</span>
      <span>${escapeHtml(payload.dataset_path ?? "")}</span>
      <span>Top K：${escapeHtml(payload.top_k ?? "")}</span>
    </div>
  `;
}

async function loadComparisonReport() {
  try {
    const response = await fetch("/evaluation/comparison-report");
    const payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(JSON.stringify(payload));
    }

    renderComparisonReport(payload);
  } catch (error) {
    setStatus(elements.reportState, "失败", "is-error");
    elements.comparisonReport.innerHTML = `<p class="muted">${escapeHtml(
      error instanceof Error ? error.message : String(error),
    )}</p>`;
  }
}

function metricStatus(value, threshold) {
  const score = Number(value);
  const target = Number(threshold);

  if (Number.isNaN(score) || Number.isNaN(target)) {
    return "is-unknown";
  }

  return score >= target ? "is-pass" : "is-warn";
}

function renderRagasMetrics(ragas) {
  if (!ragas || !ragas.metrics) {
    elements.ragasMetrics.innerHTML = "<span>最新报告没有 RAGAS 指标</span>";
    return;
  }

  const metrics = ragas.metrics;
  const thresholds = ragas.target_thresholds ?? {};
  const labels = [
    ["faithfulness", "Faithfulness"],
    ["answer_relevancy", "Answer Relevancy"],
    ["context_precision", "Context Precision"],
    ["context_recall", "Context Recall"],
  ];

  elements.ragasMetrics.innerHTML = labels
    .map(([key, label]) => {
      const value = metrics[key];
      const threshold = thresholds[key];
      const status = metricStatus(value, threshold);
      const thresholdText = Number.isFinite(Number(threshold)) ? `目标 ${formatPercent(threshold)}` : "无目标";

      return `
        <div class="ragas-score ${status}">
          <strong>${formatPercent(value)}</strong>
          <span>${escapeHtml(label)}</span>
          <small>${escapeHtml(thresholdText)}</small>
        </div>
      `;
    })
    .join("");
}

function renderRagasMeta(payload) {
  const parts = [];

  if (payload.report_path) {
    parts.push(`报告：${payload.report_path}`);
  }
  if (payload.generated_at) {
    parts.push(`生成：${formatDateTime(payload.generated_at)}`);
  }
  if (payload.summary?.cases !== undefined) {
    parts.push(`样本：${payload.summary.cases}`);
  }
  if (payload.ragas?.judge_config?.judge_model) {
    parts.push(`评审模型：${payload.ragas.judge_config.judge_model}`);
  }

  elements.ragasReportMeta.innerHTML = parts.map((part) => `<span>${escapeHtml(part)}</span>`).join("");
}

async function loadRagasReport() {
  elements.ragasButton.disabled = true;
  setStatus(elements.ragasState, "加载中");
  elements.ragasMetrics.innerHTML = "<span>正在读取报告...</span>";
  elements.ragasReportMeta.textContent = "";

  try {
    const response = await fetch("/evaluation/answer-report");
    const payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(JSON.stringify(payload));
    }

    if (!payload.available || !payload.ragas) {
      setStatus(elements.ragasState, "无 RAGAS", "is-error");
      elements.ragasMetrics.innerHTML = `<span>${escapeHtml(payload.error ?? "未找到 RAGAS 报告")}</span>`;
      renderRagasMeta(payload);
      return;
    }

    renderRagasMetrics(payload.ragas);
    renderRagasMeta(payload);
    setStatus(elements.ragasState, "已加载", "is-ok");
  } catch (error) {
    setStatus(elements.ragasState, "失败", "is-error");
    elements.ragasMetrics.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    elements.ragasButton.disabled = false;
  }
}

function sourceTags(source) {
  const tags = [];
  if (source.page !== null) {
    tags.push(`page ${source.page}`);
  }
  if (source.chunk_index !== null) {
    tags.push(`chunk ${source.chunk_index}`);
  }
  if (source.matched_child_chunk_index !== null) {
    tags.push(`child ${source.matched_child_chunk_index}`);
  }
  if (source.content_type) {
    tags.push(source.content_type);
  }
  if (source.table_index !== null) {
    tags.push(`table ${source.table_index}`);
  }
  return tags;
}

function renderSources(sources) {
  const normalized = normalizeSources(sources);
  elements.sourceCount.textContent = `${normalized.length} 条`;

  if (normalized.length === 0) {
    elements.sourcesList.innerHTML = '<p class="muted">没有引用来源。</p>';
    return;
  }

  elements.sourcesList.innerHTML = normalized
    .map((source, index) => {
      const tags = sourceTags(source);

      return `
        <article class="source-card">
          <div class="source-card-top">
            <div>
              <strong>${escapeHtml(basename(source.source) || `source ${index + 1}`)}</strong>
              <span>${escapeHtml(source.source)}</span>
            </div>
            <code>#${escapeHtml(index + 1)}</code>
          </div>
          <div class="source-meta">
            ${tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("") || "<span>metadata missing</span>"}
          </div>
          <p class="source-content">${escapeHtml(source.content)}</p>
        </article>
      `;
    })
    .join("");
}

function compactContent(value, maxLength = 180) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();

  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength)}...`;
}

function renderCandidateRows(candidates, emptyText) {
  if (!Array.isArray(candidates) || candidates.length === 0) {
    return `<p class="muted">${escapeHtml(emptyText)}</p>`;
  }

  return candidates
    .slice(0, 8)
    .map((candidate) => {
      const score = Number.isFinite(Number(candidate.score)) ? Number(candidate.score).toFixed(4) : "n/a";
      const chunk = candidate.chunk_index === null || candidate.chunk_index === undefined ? "chunk n/a" : `chunk ${candidate.chunk_index}`;
      const contentType = candidate.content_type ? ` · ${candidate.content_type}` : "";
      const variant = candidate.query_variant ? `<small>${escapeHtml(candidate.query_variant)}</small>` : "";

      return `
        <article class="debug-row">
          <div>
            <strong>${escapeHtml(candidate.source ?? "")}</strong>
            <span>${escapeHtml(chunk)}${escapeHtml(contentType)}</span>
          </div>
          <code>${escapeHtml(score)}</code>
          ${variant}
          <p>${escapeHtml(compactContent(candidate.content))}</p>
        </article>
      `;
    })
    .join("");
}

function renderRetrievalDebug(trace) {
  if (!trace || typeof trace !== "object") {
    elements.debugState.textContent = "无 trace";
    elements.debugContent.innerHTML = '<p class="muted">本次响应没有返回检索调试信息。</p>';
    return;
  }

  const variants = Array.isArray(trace.query_variants) ? trace.query_variants : [];
  const parentHydration = Array.isArray(trace.parent_hydration) ? trace.parent_hydration : [];
  const hydratedCount = parentHydration.filter((item) => item.status === "parent_hydrated").length;
  elements.debugState.textContent = `${variants.length} variants · ${hydratedCount} parents`;
  elements.debugContent.innerHTML = `
    <section class="debug-section">
      <h3>Query Variants</h3>
      <ol>${variants.map((variant) => `<li>${escapeHtml(variant)}</li>`).join("") || "<li>无</li>"}</ol>
    </section>
    <section class="debug-section">
      <h3>Dense Candidates</h3>
      ${renderCandidateRows(trace.dense_candidates, "没有 dense 候选。")}
    </section>
    <section class="debug-section">
      <h3>BM25 Candidates</h3>
      ${renderCandidateRows(trace.bm25_candidates, "没有 BM25 候选。")}
    </section>
    <section class="debug-section">
      <h3>RRF Scores</h3>
      ${renderCandidateRows(trace.rrf_scores, "没有 RRF 分数。")}
    </section>
    <section class="debug-section">
      <h3>Reranker Scores</h3>
      ${renderCandidateRows(trace.reranker_scores, "没有 reranker 分数。")}
    </section>
    <section class="debug-section">
      <h3>Parent Hydration</h3>
      <p class="muted">${escapeHtml(hydratedCount)} 个候选被替换为父块，${escapeHtml(parentHydration.length)} 条处理记录。</p>
    </section>
  `;
}

function setFeedbackState(text, mode = "") {
  elements.feedbackState.textContent = text;
  elements.feedbackState.classList.remove("is-ok", "is-error", "is-warn");
  if (mode) {
    elements.feedbackState.classList.add(mode);
  }
}

function setFeedbackEnabled(enabled) {
  elements.feedbackGoodButton.disabled = !enabled;
  elements.feedbackBadButton.disabled = !enabled;
}

function resetFeedback() {
  currentSessionId = null;
  setFeedbackEnabled(false);
  setFeedbackState("等待审计记录");
}

async function submitFeedback(rating, tags) {
  if (!currentSessionId) {
    setFeedbackState("无可反馈记录", "is-warn");
    return;
  }

  setFeedbackEnabled(false);
  setFeedbackState("提交中");

  try {
    const response = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: currentSessionId,
        rating,
        tags,
      }),
    });
    const payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(JSON.stringify(payload));
    }

    setFeedbackState("已记录", "is-ok");
  } catch (error) {
    setFeedbackEnabled(true);
    setFeedbackState("提交失败", "is-error");
    console.error(error);
  }
}

function handleSseFrame(frame) {
  if (frame.event === "token") {
    if (!streamHasToken) {
      elements.answerOutput.textContent = "";
      elements.answerOutput.classList.remove("is-loading");
      streamHasToken = true;
    }
    elements.answerOutput.textContent += frame.data;
    return;
  }

  if (frame.event === "sources") {
    try {
      renderSources(JSON.parse(frame.data));
    } catch {
      elements.sourcesList.innerHTML = `<p class="muted">${escapeHtml(frame.data)}</p>`;
    }
    return;
  }

  if (frame.event === "session") {
    try {
      const payload = JSON.parse(frame.data);
      currentSessionId = payload.session_id || null;
      setFeedbackEnabled(Boolean(currentSessionId));
      setFeedbackState(currentSessionId ? "可反馈" : "无审计记录", currentSessionId ? "is-ok" : "is-warn");
    } catch {
      setFeedbackState("审计记录解析失败", "is-error");
    }
    return;
  }

  if (frame.event === "debug") {
    try {
      renderRetrievalDebug(JSON.parse(frame.data));
    } catch {
      elements.debugState.textContent = "trace 解析失败";
      elements.debugContent.textContent = frame.data;
    }
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
  streamHasToken = false;
  elements.askButton.disabled = true;
  elements.answerOutput.classList.remove("is-error");
  elements.answerOutput.classList.add("is-loading");
  elements.answerOutput.innerHTML = `
    <div class="answer-skeleton"></div>
    <div class="answer-skeleton short"></div>
    <div class="answer-skeleton"></div>
  `;
  renderSources([]);
  renderRetrievalDebug(null);
  resetFeedback();
  setStatus(elements.answerState, "生成中");

  try {
    const requestBody = {
      question,
      top_k: topK,
      debug: true,
      rewrite_enabled: elements.rewriteToggle.checked,
      hyde_enabled: elements.hydeToggle.checked,
      parent_hydration_enabled: elements.parentToggle.checked,
    };

    const response = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
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

    if (!streamHasToken) {
      elements.answerOutput.textContent = "模型未返回内容。";
      elements.answerOutput.classList.remove("is-loading");
    }
    setStatus(elements.answerState, "完成", "is-ok");
  } catch (error) {
    if (error.name !== "AbortError") {
      setStatus(elements.answerState, "失败", "is-error");
      elements.answerOutput.classList.remove("is-loading");
      elements.answerOutput.classList.add("is-error");
      elements.answerOutput.textContent = error instanceof Error ? error.message : String(error);
    }
  } finally {
    elements.askButton.disabled = false;
    currentController = null;
  }
}

elements.ingestButton.addEventListener("click", ingestDocuments);
elements.evaluationButton.addEventListener("click", runEvaluation);
elements.ragasButton.addEventListener("click", loadRagasReport);
elements.feedbackGoodButton.addEventListener("click", () => submitFeedback(1, ["helpful"]));
elements.feedbackBadButton.addEventListener("click", () => submitFeedback(-1, ["not_accurate"]));

elements.presetQuestions.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-question]");
  if (!button) {
    return;
  }

  elements.questionInput.value = button.dataset.question;
  elements.questionInput.focus();
});

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
  elements.answerOutput.classList.remove("is-loading", "is-error");
  elements.answerOutput.textContent = "答案会实时显示在这里。";
  renderSources([]);
  renderRetrievalDebug(null);
  resetFeedback();
  setStatus(elements.answerState, "等待问题");
});

renderPresets();
checkHealth();
loadCorpusStatus();
loadComparisonReport();
loadRagasReport();
