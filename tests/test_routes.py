"""
Route-level tests: HTTP status codes, schema validation, auth enforcement.
No real Ollama/Chroma calls — engines and clients are swapped via
app.dependency_overrides, per the checklist's "Route tests" tier.
"""
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings, Settings
from app.core.ingest_engine import IngestResult
from app.core.rag_engine import RagAnswer
from app.deps import (
    get_ingest_engine,
    get_job_tracker,
    get_ollama_client,
    get_rag_engine,
    get_vector_store,
    require_api_key,
)
from app.jobs.job_tracker import JobTracker
from app.main import app

API_KEY = "test-key-123"


@pytest.fixture(autouse=True)
def override_settings():
    def _settings():
        return Settings(API_KEY=API_KEY, ALLOWED_ORIGINS="http://localhost:5173")

    app.dependency_overrides[get_settings] = _settings
    # Route tests must never build the real Chroma/Ollama clients: on a CI
    # runner there is no /data folder, and these tests must not need one.
    app.dependency_overrides[get_vector_store] = lambda: FakeStore()
    app.dependency_overrides[get_ollama_client] = lambda: FakeOllama()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


class FakeOllama:
    def is_reachable(self):
        return True


class FakeStore:
    def is_reachable(self):
        return True

    def get_all_sources(self):
        return {
            "report.txt": {
                "chunk_count": 3,
                "file_hash": "abc123",
                "mtime": "1700000000",
                "last_indexed": "2026-01-01T00:00:00+00:00",
            }
        }

    def delete_by_source(self, source_path):
        return None


class FakeRagEngine:
    def answer(self, question: str):
        return RagAnswer(answer="42", sources=[], grounded=True)


class FakeJobTracker:
    """Records what the background ingest job does to it."""

    def __init__(self):
        self.events = []

    def create_job(self):
        return "job-abc"

    def mark_running(self, job_id):
        self.events.append(("running", job_id))

    def mark_completed(self, job_id, **kwargs):
        self.events.append(("completed", job_id, kwargs["files_indexed"]))

    def mark_failed(self, job_id, error):
        self.events.append(("failed", job_id, error))

    def get(self, job_id):
        return None


class FakeIngestEngine:
    def __init__(self, error=None):
        self.error = error

    def run(self):
        if self.error:
            raise RuntimeError(self.error)
        return IngestResult(files_scanned=2, files_indexed=1, files_skipped=1, files_unchanged=0)


def _override_auth_ok():
    app.dependency_overrides[require_api_key] = lambda: None


def _clear_overrides():
    app.dependency_overrides.clear()


# ---------- /health (no auth required) ----------

def test_health_ok(client):
    app.dependency_overrides[get_ollama_client] = lambda: FakeOllama()
    app.dependency_overrides[get_vector_store] = lambda: FakeStore()

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    _clear_overrides()


# ---------- auth enforcement ----------

def test_query_requires_api_key(client):
    resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 401


def test_query_rejects_wrong_api_key(client):
    resp = client.post(
        "/query", json={"question": "hi"}, headers={"X-API-Key": "wrong-key"}
    )
    assert resp.status_code == 401


# ---------- /query ----------

def test_query_returns_grounded_response(client):
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_rag_engine] = lambda: FakeRagEngine()

    resp = client.post(
        "/query", json={"question": "What is the answer?"}, headers={"X-API-Key": API_KEY}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["answer"] == "42"
    _clear_overrides()


def test_query_rejects_empty_question(client):
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_rag_engine] = lambda: FakeRagEngine()

    resp = client.post("/query", json={"question": ""}, headers={"X-API-Key": API_KEY})

    assert resp.status_code == 422  # pydantic min_length=1 validation
    _clear_overrides()


# ---------- /sources ----------

def test_list_sources(client):
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_vector_store] = lambda: FakeStore()

    resp = client.get("/sources", headers={"X-API-Key": API_KEY})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_path"] == "report.txt"
    _clear_overrides()


# ---------- /ingest ----------

def test_trigger_ingest_returns_job_id_and_runs_the_job(client):
    tracker = FakeJobTracker()
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_ingest_engine] = lambda: FakeIngestEngine()
    app.dependency_overrides[get_job_tracker] = lambda: tracker

    resp = client.post("/ingest", headers={"X-API-Key": API_KEY})

    assert resp.status_code == 200
    assert resp.json()["job_id"] == "job-abc"
    # The test client finishes background tasks before returning.
    assert tracker.events == [("running", "job-abc"), ("completed", "job-abc", 1)]
    _clear_overrides()


def test_failed_ingest_job_is_recorded_as_failed(client):
    tracker = FakeJobTracker()
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_ingest_engine] = lambda: FakeIngestEngine(error="boom")
    app.dependency_overrides[get_job_tracker] = lambda: tracker

    resp = client.post("/ingest", headers={"X-API-Key": API_KEY})

    assert resp.status_code == 200
    assert tracker.events == [("running", "job-abc"), ("failed", "job-abc", "boom")]
    _clear_overrides()


def test_ingest_status_404_for_unknown_job(client):
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_job_tracker] = lambda: JobTracker()

    resp = client.get("/ingest/status/does-not-exist", headers={"X-API-Key": API_KEY})

    assert resp.status_code == 404
    _clear_overrides()
