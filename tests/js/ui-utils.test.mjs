import assert from "node:assert/strict";
import test from "node:test";

import {
  escapeHtml,
  parseSseChunk,
  normalizeSources,
  clampTopK,
  formatPercent,
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
      content: "RAG 内容",
    },
  ]);
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
