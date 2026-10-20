"""
Chroma wrapper. Implements VectorStore from interfaces.py. No other module
should import chromadb directly.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings


class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str):
        # Chroma sends anonymous usage telemetry (to PostHog) by default.
        # This project is local-only, so it is disabled explicitly.
        self._client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        # Chroma defaults to squared-L2 distance. The similarity threshold in
        # config.py is defined on cosine distance, so set the space explicitly.
        # NOTE: this only applies when the collection is first created; an
        # existing collection keeps its original space (see README).
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ---- VectorStore ----

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        out = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                }
            )
        return out

    def delete_by_source(self, source_path: str) -> None:
        self._collection.delete(where={"source_path": source_path})

    def get_all_sources(self) -> dict[str, dict]:
        """Chroma has no native GROUP BY, so this pulls all metadata and
        aggregates in Python. Fine at the scale this project targets
        (a personal/small-team document directory); revisit if the corpus
        grows to tens of thousands of chunks."""
        if self._collection.count() == 0:
            return {}
        result = self._collection.get(include=["metadatas"])
        sources: dict[str, dict] = {}
        for meta in result.get("metadatas", []):
            src = meta.get("source_path")
            if src is None:
                continue
            if src not in sources:
                sources[src] = {
                    "chunk_count": 0,
                    "file_hash": meta.get("file_hash", ""),
                    "mtime": meta.get("mtime", ""),
                    "last_indexed": meta.get("last_indexed", ""),
                    "index_fingerprint": meta.get("index_fingerprint", ""),
                }
            sources[src]["chunk_count"] += 1
        return sources

    def is_reachable(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:
            return False
