from pathlib import Path
from uuid import uuid4

import chromadb

from app.ingestion.chunker import Chunk


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embeddings):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)
        ids = [str(uuid4()) for _ in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
        return len(chunks)

    def similarity_search(self, query: str, top_k: int) -> list[Chunk]:
        vector = self.embeddings.embed_query(query)
        result = self.collection.query(query_embeddings=[vector], n_results=top_k)

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        chunks: list[Chunk] = []

        for content, metadata in zip(documents, metadatas, strict=False):
            metadata = metadata or {}
            source = str(metadata.get("source", ""))
            chunks.append(Chunk(content=content, source=source, metadata=dict(metadata)))

        return chunks
