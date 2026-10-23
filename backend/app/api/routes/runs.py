"""External runs: let the user's own training scripts stream into NeuroVision.

Follow a run's events with the same WebSocket as built-in jobs:
WS /api/v1/training/{run_id}/events.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.api.routes.models import Repository
from app.core.errors import ApiError, invalid, not_found
from app.schemas.errors import ErrorResponse
from app.schemas.events import RunCommand
from app.schemas.runs import (
    CommandAck,
    CommandList,
    CommandRequest,
    EventBatch,
    IngestResult,
    RunCreate,
    RunCreated,
    RunFinish,
    RunInfo,
    RunList,
)
from app.services.external_runs import ExternalRun, ExternalRunManager

router = APIRouter(prefix="/runs", tags=["runs"])

RunId = Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
# A full batch (32 steps x 8 layers x 256 values) is well under this.
MAX_BODY_BYTES = 2 * 1024 * 1024
ERRORS = {
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


def get_runs(request: Request) -> ExternalRunManager:
    runs: ExternalRunManager = request.app.state.external_runs
    return runs


Runs = Annotated[ExternalRunManager, Depends(get_runs)]


def _too_large() -> ApiError:
    return ApiError(
        413,
        "payload_too_large",
        f"Event batches can be at most {MAX_BODY_BYTES // (1024 * 1024)} MB. "
        "Send fewer steps per request.",
    )


async def _read_batch(request: Request) -> bytes:
    """Read the body, refusing it as soon as it exceeds MAX_BODY_BYTES."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise _too_large()
    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_BODY_BYTES:
            raise _too_large()
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_batch(raw: bytes) -> EventBatch:
    try:
        return EventBatch.model_validate_json(raw)
    except ValidationError as error:
        # Same envelope and field paths as FastAPI's own body validation.
        raise RequestValidationError(
            [{**e, "loc": ("body", *e["loc"])} for e in error.errors()]
        ) from None


def _writable(
    runs: ExternalRunManager, run_id: str, authorization: str | None
) -> ExternalRun:
    run = runs.get(run_id)
    if run is None:
        raise not_found(f"Run '{run_id}' was not found. It may have expired.")
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ApiError(
            401,
            "unauthorized",
            "Send the run's write token as 'Authorization: Bearer <token>'.",
        )
    if not run.token_matches(token.strip()):
        raise ApiError(403, "forbidden", "The write token does not match this run.")
    return run


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=RunCreated,
    responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
def create_run(body: RunCreate, runs: Runs, repository: Repository) -> RunCreated:
    """Start an external run. The write token is returned only here."""
    if body.model_id is not None and repository.get(body.model_id) is None:
        raise invalid("body.model_id", f"Model '{body.model_id}' was not found.")
    run, token = runs.create(body)
    return RunCreated(run=run.info(), write_token=token)


@router.get("", response_model=RunList)
def list_runs(runs: Runs) -> RunList:
    return RunList(runs=[run.info() for run in runs.list()])


@router.get("/{run_id}", response_model=RunInfo, responses={404: ERRORS[404]})
def get_run(run_id: RunId, runs: Runs) -> RunInfo:
    runs.expire_idle()
    run = runs.get(run_id)
    if run is None:
        raise not_found(f"Run '{run_id}' was not found. It may have expired.")
    return run.info()


@router.post(
    "/{run_id}/events",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestResult,
    responses=ERRORS,
)
async def log_events(
    run_id: RunId,
    request: Request,
    runs: Runs,
    authorization: Annotated[str | None, Header()] = None,
) -> IngestResult:
    """Append logged steps (an EventBatch body); an empty batch is a heartbeat.

    The token is checked before the body is read, and the whole batch is
    validated before any of it is published.
    """
    run = _writable(runs, run_id, authorization)
    raw = await _read_batch(request)

    def ingest() -> int:
        batch = _parse_batch(raw)
        run.expire_if_idle(runs.now())
        return run.ingest(batch.events, runs.now())

    accepted = await run_in_threadpool(ingest)
    return IngestResult(
        accepted=accepted,
        steps_logged=run.steps_logged,
        commands=run.pending_commands(),
    )


@router.post("/{run_id}/finish", response_model=RunInfo, responses=ERRORS)
def finish_run(
    run_id: RunId,
    body: RunFinish,
    runs: Runs,
    authorization: Annotated[str | None, Header()] = None,
) -> RunInfo:
    run = _writable(runs, run_id, authorization)
    run.finish(body, runs.now())
    return run.info()


# Two-way control (ADR 0008) -------------------------------------------------


@router.post(
    "/{run_id}/commands",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunCommand,
    responses={**ERRORS, 429: {"model": ErrorResponse}},
)
def issue_command(run_id: RunId, body: CommandRequest, runs: Runs) -> RunCommand:
    """Ask the script to pause, resume, stop, checkpoint, or change its learning rate.

    The script applies the command at its next run.poll() and acknowledges
    it; follow the outcome as run.command events on the run's WebSocket.
    """
    runs.expire_idle()
    run = runs.get(run_id)
    if run is None:
        raise not_found(f"Run '{run_id}' was not found. It may have expired.")
    return run.issue(body, runs.now())


@router.get("/{run_id}/commands", response_model=CommandList, responses=ERRORS)
def pending_commands(
    run_id: RunId,
    runs: Runs,
    authorization: Annotated[str | None, Header()] = None,
) -> CommandList:
    """Pending commands, for the script (write token required)."""
    run = _writable(runs, run_id, authorization)
    return CommandList(commands=run.pending_commands())


@router.post(
    "/{run_id}/commands/{command_id}/ack",
    response_model=RunCommand,
    responses=ERRORS,
)
def acknowledge_command(
    run_id: RunId,
    command_id: Annotated[int, Path(ge=1)],
    body: CommandAck,
    runs: Runs,
    authorization: Annotated[str | None, Header()] = None,
) -> RunCommand:
    """The script reports that it applied or rejected a command."""
    run = _writable(runs, run_id, authorization)
    return run.acknowledge(command_id, body, runs.now())
