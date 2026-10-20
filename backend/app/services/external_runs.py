"""External runs: replayable event streams fed by the user's own scripts.

A script creates a run, receives a write token once, and posts batches of
logged steps. Each run is an EventStream, so the training WebSocket replays
and follows it like a built-in job. Memory stays bounded: a run accepts at
most MAX_STEPS_PER_RUN steps, and only its latest step keeps activations in
the replay history (the UI shows only the latest anyway).
"""

import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from app.core.errors import ApiError, invalid
from app.schemas.events import (
    TrainingCompleted,
    TrainingFailed,
    TrainingStarted,
    TrainingStep,
)
from app.schemas.runs import (
    MAX_STEPS_PER_RUN,
    LoggedStep,
    RunCreate,
    RunFinish,
    RunInfo,
    RunStatus,
)
from app.services.event_stream import Event, EventStream

# A run with no events or heartbeats for this long is marked failed.
IDLE_TIMEOUT_S = 300.0
STALE_MESSAGE = (
    "No events arrived for 5 minutes, so this run was closed. "
    "The script may have stopped without calling run.finish()."
)
FAILED_MESSAGE = "The training script reported a failure."


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class ExternalRun(EventStream):
    def __init__(self, request: RunCreate, token: str, now: float) -> None:
        super().__init__()
        self.run_id = f"run_{secrets.token_hex(6)}"
        self.name = request.name
        self.model_id = request.model_id
        self.total_epochs = request.total_epochs
        self.status: RunStatus = "running"
        self.epoch = 0
        self.step: int | None = None
        self.steps_logged = 0
        self.created_at = now
        self.last_event_at = now
        self._last_metrics: dict[str, float] = {}
        self._token_digest = _digest(token)
        # Serializes ingest, finish, and expiry so checks and publishing agree.
        self.ingest_lock = threading.Lock()
        self._activation_index: int | None = None
        self.publish(
            TrainingStarted(
                **self._base,
                event_type="training.started",
                model_id=self.model_id or self.run_id,
                total_epochs=self.total_epochs,
            ).model_dump(mode="json")
        )

    @property
    def _base(self) -> dict[str, str]:
        return {"schema_version": "1.0", "job_id": self.run_id}

    @property
    def active(self) -> bool:
        return self.status == "running"

    def token_matches(self, token: str) -> bool:
        return hmac.compare_digest(self._token_digest, _digest(token))

    def info(self) -> RunInfo:
        return RunInfo(
            run_id=self.run_id,
            name=self.name,
            model_id=self.model_id,
            status=self.status,
            total_epochs=self.total_epochs,
            epoch=self.epoch,
            step=self.step or 0,
            steps_logged=self.steps_logged,
            created_at=self.created_at,
            last_event_at=self.last_event_at,
        )

    def ingest(self, steps: list[LoggedStep], now: float) -> int:
        """Validate the whole batch, then publish it; all or nothing."""
        with self.ingest_lock:
            self._require_running()
            if self.steps_logged + len(steps) > MAX_STEPS_PER_RUN:
                raise ApiError(
                    409,
                    "conflict",
                    f"This run reached its limit of {MAX_STEPS_PER_RUN} logged steps. "
                    "Log less often (for example once per epoch), or start a new run.",
                )
            previous_step, previous_epoch = self.step, self.epoch
            for i, logged in enumerate(steps):
                if previous_step is not None and logged.step <= previous_step:
                    raise invalid(
                        f"body.events.{i}.step",
                        f"Steps must increase; got {logged.step} after {previous_step}.",
                    )
                if logged.epoch < previous_epoch:
                    raise invalid(
                        f"body.events.{i}.epoch",
                        f"Epochs must not decrease; got {logged.epoch} after {previous_epoch}.",
                    )
                previous_step, previous_epoch = logged.step, logged.epoch
            for logged in steps:
                self.publish(
                    TrainingStep(
                        **self._base,
                        event_type="training.step",
                        epoch=logged.epoch,
                        step=logged.step,
                        metrics=logged.metrics,
                        selected_activations=logged.selected_activations,
                    ).model_dump(mode="json")
                )
                self.step, self.epoch = logged.step, logged.epoch
                self._last_metrics = logged.metrics
            self.steps_logged += len(steps)
            self.last_event_at = now
            return len(steps)

    def finish(self, request: RunFinish, now: float) -> None:
        with self.ingest_lock:
            self._require_running()
            self.last_event_at = now
            if request.status == "completed":
                self.status = "completed"
                self.publish(
                    TrainingCompleted(
                        **self._base,
                        event_type="training.completed",
                        epoch=self.epoch,
                        step=self.step or 0,
                        metrics=self._last_metrics,
                    ).model_dump(mode="json")
                )
            else:
                self._fail(request.message or FAILED_MESSAGE)

    def expire_if_idle(self, now: float) -> None:
        with self.ingest_lock:
            if self.active and now - self.last_event_at > IDLE_TIMEOUT_S:
                self._fail(STALE_MESSAGE)

    def _fail(self, message: str) -> None:
        self.status = "failed"
        self.publish(
            TrainingFailed(
                **self._base, event_type="training.failed", message=message
            ).model_dump(mode="json")
        )

    def _require_running(self) -> None:
        if not self.active:
            raise ApiError(
                409, "conflict", f"This run has already {self.status}. Start a new run."
            )

    def _record(self, event: Event) -> None:
        if event.get("selected_activations"):
            # Keep activations only on the latest step that has them.
            if self._activation_index is not None:
                older = self._events[self._activation_index]
                self._events[self._activation_index] = {
                    **older,
                    "selected_activations": {},
                }
            self._activation_index = len(self._events)
        self._events.append(event)


class ExternalRunManager:
    def __init__(
        self,
        max_active: int = 8,
        retained: int = 32,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._max_active = max_active
        self._retained = retained
        self._clock = clock
        self._runs: OrderedDict[str, ExternalRun] = OrderedDict()
        self._lock = threading.Lock()

    def create(self, request: RunCreate) -> tuple[ExternalRun, str]:
        self.expire_idle()
        token = secrets.token_urlsafe(32)
        with self._lock:
            if sum(run.active for run in self._runs.values()) >= self._max_active:
                raise ApiError(
                    429,
                    "too_many_requests",
                    f"{self._max_active} external runs are already active. "
                    "Finish one (run.finish()) or wait for idle runs to time out.",
                )
            run = ExternalRun(request, token, self._clock())
            self._runs[run.run_id] = run
            self._trim()
        return run, token

    def get(self, run_id: str) -> ExternalRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[ExternalRun]:
        """Newest first."""
        self.expire_idle()
        with self._lock:
            return list(reversed(self._runs.values()))

    def now(self) -> float:
        return self._clock()

    def expire_idle(self) -> None:
        """Close runs whose script went quiet; called on every API access."""
        now = self._clock()
        with self._lock:
            runs = [run for run in self._runs.values() if run.active]
        for run in runs:
            run.expire_if_idle(now)

    def _trim(self) -> None:
        """Forget the oldest finished runs past the retention limit."""
        finished = [rid for rid, run in self._runs.items() if not run.active]
        for run_id in finished[: max(0, len(self._runs) - self._retained)]:
            del self._runs[run_id]
