import json
from pathlib import Path

import chromadb

from app.ingestion.chunker import Chunk
from app.rag.documents import chunk_id
from app.security.acl import RetrievalAccessFilter


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
        ids = [chunk_id(chunk) for chunk in chunks]
        metadatas = []
        for chunk, identity in zip(chunks, ids, strict=False):
            metadata = dict(chunk.metadata)
            metadata["chunk_id"] = identity
            metadata["source"] = chunk.source
            metadatas.append(chroma_compatible_metadata(metadata))

        self.collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int,
        access_filter: RetrievalAccessFilter | None = None,
    ) -> list[Chunk]:
        vector = self.embeddings.embed_query(query)
        query_kwargs = {"query_embeddings": [vector], "n_results": top_k}
        if access_filter is not None and not access_filter.allow_missing_acl:
            query_kwargs["where"] = {"tenant_id": access_filter.tenant_id}
        result = self.collection.query(**query_kwargs)

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        chunks: list[Chunk] = []

        for content, metadata in zip(documents, metadatas, strict=False):
            metadata = metadata or {}
            if access_filter is not None and not access_filter.can_access_metadata(metadata):
                continue
            source = str(metadata.get("source", ""))
            chunks.append(Chunk(content=content, source=source, metadata=dict(metadata)))

        return chunks


def chroma_compatible_metadata(metadata: dict) -> dict:
    normalized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        elif isinstance(value, (list, tuple, set, dict)):
            normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            normalized[key] = str(value)
    return normalized
