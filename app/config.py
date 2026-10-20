"""
Env-driven settings. Nothing in this app should read os.environ directly
anywhere else — config.py is the one place that does, per the checklist's
intent of keeping model names / thresholds swappable without code changes.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Ollama ---
    ollama_host: str = Field(default="http://ollama:11434", alias="OLLAMA_HOST")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    llm_model: str = Field(default="llama3.1:8b", alias="LLM_MODEL")

    # --- Chroma ---
    chroma_dir: str = Field(default="/data/chroma", alias="CHROMA_DIR")
    chroma_collection: str = Field(default="local_rag_docs", alias="CHROMA_COLLECTION")

    # --- Ingestion ---
    docs_dir: str = Field(default="/data/docs", alias="DOCS_DIR")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")  # chars, not tokens
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    allowed_extensions: str = Field(
        default=".txt,.md,.pdf,.docx", alias="ALLOWED_EXTENSIONS"
    )

    # --- Retrieval / grounding ---
    top_k: int = Field(default=5, alias="TOP_K")
    # Chroma returns cosine DISTANCE (0 = identical, higher = less similar).
    # A query is considered "grounded" only if the closest chunk's distance
    # is <= this threshold. Tune per embedding model / corpus.
    similarity_distance_threshold: float = Field(
        default=0.45, alias="SIMILARITY_DISTANCE_THRESHOLD"
    )

    # --- API ---
    api_key: str = Field(default="changeme-local-dev-key", alias="API_KEY")
    allowed_origins: str = Field(default="http://localhost:5173", alias="ALLOWED_ORIGINS")

    class Config:
        env_file = ".env"
        populate_by_name = True

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()]

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
