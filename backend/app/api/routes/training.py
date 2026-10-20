from typing import Annotated

from fastapi import APIRouter, Path, WebSocket, WebSocketDisconnect, status

from app.api.routes.models import Jobs
from app.core.config import get_settings
from app.core.errors import not_found
from app.schemas.errors import ErrorResponse
from app.schemas.training import TrainingJobInfo
from app.services.event_stream import TERMINAL_EVENTS, EventStream
from app.services.external_runs import ExternalRunManager
from app.services.training_jobs import TrainingJobManager

router = APIRouter(prefix="/training", tags=["training"])

JobId = Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
NOT_FOUND = {404: {"model": ErrorResponse}}

# Application-defined WebSocket close codes (4000-4999).
CLOSE_UNKNOWN_JOB = 4404
CLOSE_FORBIDDEN_ORIGIN = 4403


@router.get("/{job_id}", response_model=TrainingJobInfo, responses=NOT_FOUND)
def get_job(job_id: JobId, jobs: Jobs) -> TrainingJobInfo:
    job = jobs.get(job_id)
    if job is None:
        raise not_found(f"Training job '{job_id}' was not found.")
    return job.info()


@router.post(
    "/{job_id}/cancel",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TrainingJobInfo,
    responses=NOT_FOUND,
)
def cancel_job(job_id: JobId, jobs: Jobs) -> TrainingJobInfo:
    """Request cancellation; the job stops at its next batch boundary."""
    job = jobs.cancel(job_id)
    if job is None:
        raise not_found(f"Training job '{job_id}' was not found.")
    return job.info()


@router.websocket("/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    """Replay the job's events so far, then stream new ones until it ends.

    Also follows external runs (ids from POST /runs), which share the format.
    """
    # CORS does not cover WebSockets; check the browser origin explicitly.
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in get_settings().cors_origins:
        await websocket.close(code=CLOSE_FORBIDDEN_ORIGIN, reason="Origin not allowed.")
        return

    jobs: TrainingJobManager = websocket.app.state.training_jobs
    runs: ExternalRunManager = websocket.app.state.external_runs
    job: EventStream | None = jobs.get(job_id) or runs.get(job_id)
    await websocket.accept()
    if job is None:
        await websocket.close(code=CLOSE_UNKNOWN_JOB, reason="Unknown training job.")
        return

    history, queue = job.subscribe()
    try:
        for event in history:
            await websocket.send_json(event)
            if event["event_type"] in TERMINAL_EVENTS:
                await websocket.close()
                return
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event["event_type"] in TERMINAL_EVENTS:
                await websocket.close()
                return
    except WebSocketDisconnect:
        pass
    finally:
        job.unsubscribe(queue)
