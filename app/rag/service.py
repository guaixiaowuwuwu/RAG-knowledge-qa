from dataclasses import dataclass
from typing import Protocol

from app.ingestion.chunker import Chunk
from app.rag.prompts import build_rag_prompt


@dataclass(frozen=True)
class Source:
    source: str
    page: int | None
    chunk_index: int | None
    content: str


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source]


class Retriever(Protocol):
    def similarity_search(self, query: str, top_k: int) -> list[Chunk]:
        ...


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class RagService:
    def __init__(self, retriever: Retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, top_k: int) -> Answer:
        chunks = self.retriever.similarity_search(question, top_k=top_k)
        if not chunks:
            return Answer(answer="知识库中没有找到相关内容，无法基于现有资料回答。", sources=[])

        prompt = build_rag_prompt(question, chunks)
        answer = self.llm.complete(prompt)
        sources = [
            Source(
                source=chunk.source,
                page=chunk.metadata.get("page"),
                chunk_index=chunk.metadata.get("chunk_index"),
                content=chunk.content,
            )
            for chunk in chunks
        ]
        return Answer(answer=answer, sources=sources)
