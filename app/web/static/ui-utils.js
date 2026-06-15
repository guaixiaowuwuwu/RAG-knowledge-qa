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
    .replaceAll("\r\n", "\n")
    .replaceAll("\r", "\n")
    .split(/\n{2,}/)
    .map((frame) => frame.replace(/\n+$/, ""))
    .filter((frame) => frame.length > 0)
    .map((frame) => {
      const lines = frame.split("\n");
      let event = "message";
      const dataLines = [];

      for (const line of lines) {
        if (line.startsWith("event:")) {
          event = line.slice("event:".length).trim();
        } else if (line.startsWith("data:")) {
          const data = line.slice("data:".length);
          dataLines.push(data.startsWith(" ") ? data.slice(1) : data);
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
    matched_child_chunk_index: source.matched_child_chunk_index ?? null,
    content_type: source.content_type ?? null,
    table_index: source.table_index ?? null,
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

export function formatPercent(value) {
  const number = Number(value);

  if (Number.isNaN(number)) {
    return "0.0%";
  }

  return `${(number * 100).toFixed(1)}%`;
}

export function formatInteger(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "0";
  }

  return new Intl.NumberFormat("en-US").format(number);
}

export function formatBytes(value) {
  const bytes = Number(value);

  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }

  const precision = unitIndex === 0 ? 0 : 1;
  return `${size.toFixed(precision)} ${units[unitIndex]}`;
}

export function formatDateTime(value) {
  if (!value) {
    return "未生成";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function basename(path) {
  const parts = String(path ?? "").split(/[\\/]/).filter(Boolean);
  return parts.at(-1) ?? "";
}

export function hitLabel(hit, isNegative = false) {
  if (isNegative) {
    return hit ? "误召回" : "正确拒答";
  }

  return hit ? "命中" : "未命中";
}
