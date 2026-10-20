"""
Protocol definitions for the backend's pluggable dependencies.

Nothing in `ingest_engine.py` or `rag_engine.py` should import a concrete
client directly (OllamaClient, ChromaStore, etc). They depend on these
Protocols instead, which is what lets the engines be unit-tested with
fakes and zero Docker/Ollama/Chroma running (see checklist: "Testability
checklist").
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into a vector. Must be the SAME provider/model used for
    both ingestion and querying, or vectors won't be comparable."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question (some models want a query-specific prefix)."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks (some models want a document-specific prefix)."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Generates an answer given a prompt (already containing retrieved
    context) built by rag_engine.py."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Minimal vector store contract. Chroma is the current implementation
    (vector_store.py); swapping it out means only that file changes."""

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        ...

    def query(
        self, embedding: list[float], top_k: int
    ) -> list[dict]:
        """Returns a list of {id, document, metadata, distance} dicts,
        ordered by increasing distance (closest first)."""
        ...

    def delete_by_source(self, source_path: str) -> None:
        ...

    def get_all_sources(self) -> dict[str, dict]:
        """Returns {source_path: {chunk_count, file_hash, mtime,
        last_indexed}} for every distinct source currently indexed."""
        ...
