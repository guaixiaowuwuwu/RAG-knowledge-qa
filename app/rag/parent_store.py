import json
from pathlib import Path

from app.ingestion.chunker import Chunk
from app.security.acl import RetrievalAccessFilter


class JsonlParentStore:
    def __init__(self, path: Path):
        self.path = path
        self._parents = self._load()

    def _load(self) -> dict[str, Chunk]:
        if not self.path.exists():
            return {}

        parents: dict[str, Chunk] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = dict(row["metadata"])
            parent_id = str(metadata["parent_id"])
            parents[parent_id] = Chunk(
                content=str(row["content"]),
                source=str(row["source"]),
                metadata=metadata,
            )
        return parents

    def write(self, parents: list[Chunk]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for parent in parents:
                row = {
                    "content": parent.content,
                    "source": parent.source,
                    "metadata": parent.metadata,
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._parents = self._load()

    def get(self, parent_id: str) -> Chunk | None:
        return self._parents.get(parent_id)

    def hydrate(
        self,
        children: list[Chunk],
        access_filter: RetrievalAccessFilter | None = None,
    ) -> list[Chunk]:
        hydrated: list[Chunk] = []
        seen_parent_ids: set[str] = set()
        for child in children:
            parent_id = child.metadata.get("parent_id")
            parent = self.get(str(parent_id)) if parent_id else None
            if parent is None:
                hydrated.append(child)
                continue
            if str(parent_id) in seen_parent_ids:
                continue
            seen_parent_ids.add(str(parent_id))
            if access_filter is not None and not access_filter.can_access_metadata(parent.metadata):
                hydrated.append(child)
                continue
            metadata = dict(parent.metadata)
            metadata["matched_child_chunk_index"] = child.metadata.get("chunk_index")
            hydrated.append(Chunk(content=parent.content, source=parent.source, metadata=metadata))
        return hydrated
