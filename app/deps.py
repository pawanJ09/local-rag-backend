"""
FastAPI dependency providers. This is where concrete implementations
(OllamaClient, ChromaStore) get constructed and injected into the engines
— the one place main.py-and-friends are allowed to know about concrete
classes. Routes depend on these via Depends(), and tests override them
with fakes via app.dependency_overrides (per the checklist's "Route
tests" tier).

NOTE (flagged addition): this file wasn't itemized in the agreed backend
tree. It exists because FastAPI's dependency-injection pattern needs
somewhere to live; folding it into main.py was the alternative but would
have made main.py do two jobs (routing + composition root). Flagging
rather than silently expanding the tree.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from app.core.ollama_client import OllamaClient
from app.core.vector_store import ChromaStore
from app.core.ingest_engine import IngestEngine
from app.core.rag_engine import RagEngine
from app.jobs.job_tracker import JobTracker


@lru_cache
def get_ollama_client() -> OllamaClient:
    settings = get_settings()
    return OllamaClient(
        host=settings.ollama_host,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )


@lru_cache
def get_vector_store() -> ChromaStore:
    settings = get_settings()
    return ChromaStore(
        persist_dir=settings.chroma_dir, collection_name=settings.chroma_collection
    )


@lru_cache
def get_job_tracker() -> JobTracker:
    return JobTracker()


def get_ingest_engine(
    settings: Settings = Depends(get_settings),
    embedder: OllamaClient = Depends(get_ollama_client),
    store: ChromaStore = Depends(get_vector_store),
) -> IngestEngine:
    return IngestEngine(
        embedder=embedder,
        store=store,
        docs_dir=settings.docs_dir,
        allowed_extensions=settings.allowed_extensions_list,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        include_path_in_embedding=settings.embed_include_path,
        index_fingerprint=settings.index_fingerprint,
    )


def get_rag_engine(
    settings: Settings = Depends(get_settings),
    embedder: OllamaClient = Depends(get_ollama_client),
    llm: OllamaClient = Depends(get_ollama_client),
    store: ChromaStore = Depends(get_vector_store),
) -> RagEngine:
    return RagEngine(
        embedder=embedder,
        llm=llm,
        store=store,
        top_k=settings.top_k,
        similarity_distance_threshold=settings.similarity_distance_threshold,
        path_as_evidence=settings.path_as_evidence,
    )


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    # A missing header is treated like a wrong key: 401 right away, before any
    # other dependency (Chroma, Ollama) is built. With a required Header(...)
    # FastAPI answered 422 only after building them.
    if x_api_key is None or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key"
        )
