"""
Pydantic schemas. This file IS the API contract — UI and CLI repos build
their types against what's defined here (via /openapi.json). Changing a
field here is a cross-repo event, not a local one.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------- /health ----------

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    ollama_reachable: bool
    chroma_reachable: bool


# ---------- /query ----------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    source_path: str
    chunk_id: str
    excerpt: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    grounded: bool = Field(
        ..., description="False when retrieved context didn't clear the similarity threshold"
    )


# ---------- /ingest ----------

class IngestTriggerResponse(BaseModel):
    job_id: str
    status: str = "queued"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class IngestStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    skipped_files: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# ---------- /sources ----------

class SourceInfo(BaseModel):
    source_path: str
    chunk_count: int
    file_hash: str
    last_indexed: str


class SourcesListResponse(BaseModel):
    sources: List[SourceInfo]


# ---------- /config ----------

class ConfigResponse(BaseModel):
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_distance_threshold: float
    allowed_extensions: List[str]
