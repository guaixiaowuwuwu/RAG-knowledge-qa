import assert from "node:assert/strict";
import test from "node:test";

import {
  escapeHtml,
  parseSseChunk,
  normalizeSources,
  clampTopK,
  formatPercent,
  formatInteger,
  formatBytes,
  formatDateTime,
  groupedMetricRows,
  basename,
  hitLabel,
} from "../../app/web/static/ui-utils.js";

test("escapeHtml escapes dangerous characters", () => {
  assert.equal(
    escapeHtml('<script>alert("x")</script>'),
    "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;",
  );
});

test("parseSseChunk parses event source frames", () => {
  const frames = parseSseChunk("event: token\ndata: 你好\n\nevent: sources\ndata: []\n\n");

  assert.deepEqual(frames, [
    { event: "token", data: "你好" },
    { event: "sources", data: "[]" },
  ]);
});

test("parseSseChunk handles CRLF event source frames", () => {
  const frames = parseSseChunk("event: token\r\ndata: RAG\r\n\r\nevent: sources\r\ndata: []\r\n\r\n");

  assert.deepEqual(frames, [
    { event: "token", data: "RAG" },
    { event: "sources", data: "[]" },
  ]);
});

test("parseSseChunk preserves whitespace token data", () => {
  const frames = parseSseChunk("event: token\ndata:  \n\n");

  assert.deepEqual(frames, [{ event: "token", data: " " }]);
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
      matched_child_chunk_index: null,
      content_type: null,
      table_index: null,
      content: "RAG 内容",
    },
  ]);
});

test("normalizeSources preserves citation metadata", () => {
  const sources = normalizeSources([
    {
      source: "report.pdf",
      page: 3,
      chunk_index: 12,
      matched_child_chunk_index: 2,
      content_type: "table",
      table_index: 1,
      content: "|A|B|",
    },
  ]);

  assert.deepEqual(sources[0], {
    source: "report.pdf",
    page: 3,
    chunk_index: 12,
    matched_child_chunk_index: 2,
    content_type: "table",
    table_index: 1,
    content: "|A|B|",
  });
});

test("clampTopK keeps values within API bounds", () => {
  assert.equal(clampTopK("0"), 1);
  assert.equal(clampTopK("25"), 20);
  assert.equal(clampTopK("4"), 4);
});

test("formatPercent formats ratio values", () => {
  assert.equal(formatPercent(1), "100.0%");
  assert.equal(formatPercent(0.825), "82.5%");
});

test("formatInteger formats finite values", () => {
  assert.equal(formatInteger(5171), "5,171");
  assert.equal(formatInteger("bad"), "0");
});

test("formatBytes uses compact binary units", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatBytes(1048576), "1.0 MB");
});

test("formatDateTime handles empty and ISO values", () => {
  assert.equal(formatDateTime(""), "未生成");
  assert.match(formatDateTime("2026-06-14T15:00:00Z"), /06\/14|06-14|14\/06/);
});

test("basename extracts file name from source paths", () => {
  assert.equal(basename("data/documents/sec_filings/AAPL/report.htm"), "report.htm");
  assert.equal(basename(""), "");
});

test("hitLabel distinguishes positive and negative cases", () => {
  assert.equal(hitLabel(true, false), "命中");
  assert.equal(hitLabel(false, false), "未命中");
  assert.equal(hitLabel(false, true), "正确拒答");
  assert.equal(hitLabel(true, true), "误召回");
});

test("groupedMetricRows flattens grouped summaries for compact tables", () => {
  const rows = groupedMetricRows({
    language: {
      zh: { cases: 2, hit_rate_at_k: 1, negative_rejection_rate: 0 },
      en: { cases: 1, hit_rate_at_k: 0.5, negative_rejection_rate: 1 },
    },
    is_negative: {
      false: { cases: 2, hit_rate_at_k: 0.75 },
    },
  });

  assert.deepEqual(rows, [
    { dimension: "language", label: "zh", summary: { cases: 2, hit_rate_at_k: 1, negative_rejection_rate: 0 } },
    { dimension: "language", label: "en", summary: { cases: 1, hit_rate_at_k: 0.5, negative_rejection_rate: 1 } },
    { dimension: "is_negative", label: "false", summary: { cases: 2, hit_rate_at_k: 0.75 } },
  ]);
});
