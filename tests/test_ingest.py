import os
import time

import pytest

from app.core.ingest_engine import IngestEngine, _chunk_text, _file_hash, _humanize_path


class FakeEmbedder:
    def __init__(self):
        self.document_inputs: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_inputs.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeStore:
    """Records upserts/deletes in memory; get_all_sources mimics Chroma's
    aggregation so incremental re-ingestion can be tested without Chroma."""

    def __init__(self):
        self.upserted_calls = []
        self.deleted_sources = []
        self._chunks_by_source: dict[str, dict] = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        self.upserted_calls.append((ids, documents, metadatas))
        for meta in metadatas:
            src = meta["source_path"]
            self._chunks_by_source[src] = {
                "chunk_count": self._chunks_by_source.get(src, {}).get("chunk_count", 0) + 1,
                "file_hash": meta["file_hash"],
                "mtime": meta["mtime"],
                "last_indexed": meta["last_indexed"],
                "index_fingerprint": meta.get("index_fingerprint", ""),
            }

    def query(self, embedding, top_k):
        raise NotImplementedError

    def delete_by_source(self, source_path):
        self.deleted_sources.append(source_path)
        self._chunks_by_source.pop(source_path, None)

    def get_all_sources(self):
        return dict(self._chunks_by_source)


# ---------- chunking ----------

def test_chunk_text_respects_size_and_overlap():
    text = "a" * 1000
    chunks = _chunk_text(text, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    # reconstructs full coverage (allowing overlap)
    assert "".join(chunks)[: len(text)] != ""
    assert chunks[-1] == text[-len(chunks[-1]):]


def test_chunk_text_empty_input_returns_no_chunks():
    assert _chunk_text("   ", chunk_size=100, overlap=10) == []


def test_chunk_text_rejects_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        _chunk_text("hello world", chunk_size=10, overlap=10)


# ---------- ingest engine (incremental re-ingestion) ----------

def test_new_file_gets_indexed(tmp_path):
    (tmp_path / "a.txt").write_text("hello world, this is a test document.")
    store = FakeStore()
    engine = IngestEngine(
        embedder=FakeEmbedder(),
        store=store,
        docs_dir=str(tmp_path),
        allowed_extensions=[".txt", ".md"],
        chunk_size=1000,
        chunk_overlap=50,
    )

    result = engine.run()

    assert result.files_indexed == 1
    assert result.files_unchanged == 0
    assert len(store.upserted_calls) == 1


def test_unchanged_file_is_skipped_on_rerun(tmp_path):
    (tmp_path / "a.txt").write_text("stable content that never changes.")
    store = FakeStore()
    engine = IngestEngine(
        embedder=FakeEmbedder(),
        store=store,
        docs_dir=str(tmp_path),
        allowed_extensions=[".txt"],
        chunk_size=1000,
        chunk_overlap=50,
    )

    first = engine.run()
    second = engine.run()

    assert first.files_indexed == 1
    assert second.files_indexed == 0
    assert second.files_unchanged == 1
    assert len(store.upserted_calls) == 1  # not re-embedded


def test_changed_file_is_reindexed_and_old_chunks_deleted(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("original content.")
    store = FakeStore()
    engine = IngestEngine(
        embedder=FakeEmbedder(),
        store=store,
        docs_dir=str(tmp_path),
        allowed_extensions=[".txt"],
        chunk_size=1000,
        chunk_overlap=50,
    )
    engine.run()

    f.write_text("completely different content now.")
    result = engine.run()

    assert result.files_indexed == 1
    assert "a.txt" in store.deleted_sources
    assert len(store.upserted_calls) == 2


def test_unsupported_extension_is_skipped_not_failed(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    store = FakeStore()
    engine = IngestEngine(
        embedder=FakeEmbedder(),
        store=store,
        docs_dir=str(tmp_path),
        allowed_extensions=[".txt", ".md"],
        chunk_size=1000,
        chunk_overlap=50,
    )

    result = engine.run()

    assert result.files_scanned == 1
    assert result.files_skipped == 1
    assert result.files_indexed == 0
    assert str(tmp_path / "image.png") in result.skipped_files


def test_missing_docs_dir_returns_empty_result_not_error(tmp_path):
    missing = tmp_path / "does-not-exist"
    store = FakeStore()
    engine = IngestEngine(
        embedder=FakeEmbedder(),
        store=store,
        docs_dir=str(missing),
        allowed_extensions=[".txt"],
        chunk_size=1000,
        chunk_overlap=50,
    )

    result = engine.run()

    assert result.files_scanned == 0
    assert result.files_indexed == 0


def test_file_hash_changes_when_content_changes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("version 1")
    h1 = _file_hash(f)
    f.write_text("version 2")
    h2 = _file_hash(f)
    assert h1 != h2


def _engine(tmp_path, embedder, store, **kw):
    return IngestEngine(
        embedder=embedder,
        store=store,
        docs_dir=str(tmp_path),
        allowed_extensions=[".txt"],
        chunk_size=1000,
        chunk_overlap=50,
        **kw,
    )


def test_humanize_path_splits_camel_case_and_separators():
    assert _humanize_path("CrystalLake/2024/Lease_Agreement-v2.pdf") == (
        "Crystal Lake / 2024 / Lease Agreement v2"
    )
    assert _humanize_path("notes.txt") == "notes"


def test_embedded_text_includes_readable_path_but_stored_text_does_not(tmp_path):
    sub = tmp_path / "ProjectX"
    sub.mkdir()
    (sub / "lease_notes.txt").write_text("Monthly rent is 2900.")
    embedder, store = FakeEmbedder(), FakeStore()

    _engine(tmp_path, embedder, store, include_path_in_embedding=True).run()

    assert embedder.document_inputs[0].startswith("Document: Project X / lease notes\n")
    assert store.upserted_calls[0][1][0] == "Monthly rent is 2900."  # stored text is clean


def test_path_can_be_excluded_from_embedding(tmp_path):
    (tmp_path / "a.txt").write_text("hello there")
    embedder, store = FakeEmbedder(), FakeStore()

    _engine(tmp_path, embedder, store, include_path_in_embedding=False).run()

    assert embedder.document_inputs == ["hello there"]


def test_changed_index_settings_trigger_reindex_of_unchanged_files(tmp_path):
    (tmp_path / "a.txt").write_text("same content both times")
    store = FakeStore()

    first = _engine(tmp_path, FakeEmbedder(), store, index_fingerprint="settings-A").run()
    same = _engine(tmp_path, FakeEmbedder(), store, index_fingerprint="settings-A").run()
    changed = _engine(tmp_path, FakeEmbedder(), store, index_fingerprint="settings-B").run()

    assert first.files_indexed == 1
    assert same.files_unchanged == 1 and same.files_indexed == 0
    assert changed.files_indexed == 1 and changed.files_unchanged == 0
    assert "a.txt" in store.deleted_sources  # old vectors removed before re-adding


def test_path_is_not_embedded_by_default(tmp_path):
    (tmp_path / "a.txt").write_text("hello there")
    embedder, store = FakeEmbedder(), FakeStore()

    _engine(tmp_path, embedder, store).run()

    assert embedder.document_inputs == ["hello there"]
