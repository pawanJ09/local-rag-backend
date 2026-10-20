"""
In-memory job status store for async /ingest. Deliberately simple:
no persistence across backend restarts, no retry logic. Flagged nuance —
if ingestion needs to survive a restart or be queryable after a crash,
this should move to a file-backed or Redis-backed store; not needed at
this project's current scale (a single local backend process).
"""
from __future__ import annotations

import threading
import uuid

from app.models import IngestStatusResponse, JobStatus


class JobTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestStatusResponse] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = IngestStatusResponse(job_id=job_id, status=JobStatus.queued)
        return job_id

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = JobStatus.running

    def mark_completed(
        self,
        job_id: str,
        files_scanned: int,
        files_indexed: int,
        files_skipped: int,
        files_unchanged: int,
        skipped_files: list[str],
    ) -> None:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.completed
                job.files_scanned = files_scanned
                job.files_indexed = files_indexed
                job.files_skipped = files_skipped
                job.files_unchanged = files_unchanged
                job.skipped_files = skipped_files

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = JobStatus.failed
                self._jobs[job_id].error = error

    def get(self, job_id: str) -> IngestStatusResponse | None:
        with self._lock:
            return self._jobs.get(job_id)
