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
