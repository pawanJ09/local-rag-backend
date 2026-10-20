from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.core.ingest_engine import IngestEngine
from app.deps import get_ingest_engine, get_job_tracker, require_api_key
from app.jobs.job_tracker import JobTracker
from app.models import IngestStatusResponse, IngestTriggerResponse

router = APIRouter(dependencies=[Depends(require_api_key)])


def _run_ingest_job(job_id: str, engine: IngestEngine, tracker: JobTracker) -> None:
    tracker.mark_running(job_id)
    try:
        result = engine.run()
        tracker.mark_completed(
            job_id,
            files_scanned=result.files_scanned,
            files_indexed=result.files_indexed,
            files_skipped=result.files_skipped,
            files_unchanged=result.files_unchanged,
            skipped_files=result.skipped_files,
        )
    except Exception as e:  # pragma: no cover - defensive
        tracker.mark_failed(job_id, str(e))


@router.post("/ingest", response_model=IngestTriggerResponse)
def trigger_ingest(
    background_tasks: BackgroundTasks,
    engine: IngestEngine = Depends(get_ingest_engine),
    tracker: JobTracker = Depends(get_job_tracker),
) -> IngestTriggerResponse:
    job_id = tracker.create_job()
    background_tasks.add_task(_run_ingest_job, job_id, engine, tracker)
    return IngestTriggerResponse(job_id=job_id)


@router.get("/ingest/status/{job_id}", response_model=IngestStatusResponse)
def ingest_status(
    job_id: str, tracker: JobTracker = Depends(get_job_tracker)
) -> IngestStatusResponse:
    job = tracker.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job_id")
    return job
