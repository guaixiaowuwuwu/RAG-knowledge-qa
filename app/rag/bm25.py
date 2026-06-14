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

        query_tokens = tokenize(query)
        query_token_set = set(query_tokens)
        scores = self.index.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (item[1], len(query_token_set.intersection(self.tokenized_corpus[item[0]]))),
            reverse=True,
        )
        results: list[RetrievedDocument] = []
        for index, score in ranked[:top_k]:
            token_overlap = query_token_set.intersection(self.tokenized_corpus[index])
            if score <= 0 and not token_overlap:
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
