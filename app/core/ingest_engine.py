"""
Walks the docs directory, extracts text per format, chunks it, embeds it,
and upserts into the vector store. Takes EmbeddingProvider and VectorStore
via constructor injection (interfaces.py) — never imports ollama_client.py
or vector_store.py's concrete classes directly, so it can be unit-tested
with fakes.

NOTE (flagged deviation from the checklist's literal file list): per-format
text extraction (pdf/docx/txt/md) lives in this file as private functions
rather than in a separate `extractors.py`, which wasn't in the agreed
backend tree. If the format list grows (csv, images via OCR, etc.) this
should probably split out — flagging now rather than silently adding a
file the checklist didn't list.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.core.interfaces import EmbeddingProvider, VectorStore


@dataclass
class IngestResult:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    skipped_files: list[str] = field(default_factory=list)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_text(path: Path) -> str | None:
    """Returns extracted plain text, or None if the format isn't supported
    or extraction fails. Callers treat None as 'skip this file'."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            # pdfplumber keeps filled-in form values next to their labels
            # (e.g. "Monthly Rent ... 2900.00"). pypdf's default extraction
            # returned the values far from their labels, and its layout mode
            # dropped them entirely on real DocuSign leases. pypdf stays as
            # a fallback if pdfplumber fails on a file.
            try:
                import pdfplumber

                with pdfplumber.open(str(path)) as pdf:
                    return "\n".join((page.extract_text() or "") for page in pdf.pages)
            except Exception:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".docx":
            import docx

            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return None
    return None


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window chunking by character count. Simple and dependency-free;
    good enough for this project's scale. Swap for a token-aware splitter
    later if chunk boundaries start cutting mid-sentence too often."""
    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


class IngestEngine:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        docs_dir: str,
        allowed_extensions: list[str],
        chunk_size: int,
        chunk_overlap: int,
    ):
        self._embedder = embedder
        self._store = store
        self._docs_dir = Path(docs_dir)
        self._allowed_extensions = set(allowed_extensions)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def run(self) -> IngestResult:
        result = IngestResult()
        if not self._docs_dir.exists():
            return result

        existing_sources = self._store.get_all_sources()

        for path in sorted(self._docs_dir.rglob("*")):
            if not path.is_file():
                continue
            result.files_scanned += 1

            if path.suffix.lower() not in self._allowed_extensions:
                result.files_skipped += 1
                result.skipped_files.append(str(path))
                continue

            source_path = str(path.relative_to(self._docs_dir))
            current_hash = _file_hash(path)

            previously_indexed = existing_sources.get(source_path)
            if previously_indexed and previously_indexed.get("file_hash") == current_hash:
                result.files_unchanged += 1
                continue

            text = _extract_text(path)
            if text is None:
                result.files_skipped += 1
                result.skipped_files.append(str(path))
                continue

            chunks = _chunk_text(text, self._chunk_size, self._chunk_overlap)
            if not chunks:
                result.files_skipped += 1
                result.skipped_files.append(str(path))
                continue

            # Re-indexing a changed file: drop its old chunks first so
            # stale/renumbered chunks don't linger.
            if previously_indexed:
                self._store.delete_by_source(source_path)

            embeddings = self._embedder.embed_batch(chunks)
            ids = [str(uuid.uuid4()) for _ in chunks]
            mtime = str(os.path.getmtime(path))
            from datetime import datetime, timezone

            last_indexed = datetime.now(timezone.utc).isoformat()
            metadatas = [
                {
                    "source_path": source_path,
                    "chunk_index": i,
                    "file_hash": current_hash,
                    "mtime": mtime,
                    "last_indexed": last_indexed,
                }
                for i in range(len(chunks))
            ]
            self._store.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
            result.files_indexed += 1

        return result
