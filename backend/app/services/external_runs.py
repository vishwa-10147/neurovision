"""External runs: replayable event streams fed by the user's own scripts.

A script creates a run, receives a write token once, and posts batches of
logged steps. Each run is an EventStream, so the training WebSocket replays
and follows it like a built-in job. Memory stays bounded: a run accepts at
most MAX_STEPS_PER_RUN steps, and only its latest step keeps activations in
the replay history (the UI shows only the latest anyway).

With a RunStore, runs are saved to disk and reloaded at startup as read-only
replays (ADR 0013); a run still running at shutdown comes back as failed.
"""

import hashlib
import hmac
import logging
import secrets
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from app.core.errors import ApiError, invalid, not_found
from app.schemas.events import (
    RunAlert,
    RunAlertEvent,
    RunCommand,
    RunCommandEvent,
    TrainingCancelled,
    TrainingCompleted,
    TrainingFailed,
    TrainingStarted,
    TrainingStep,
)
from app.schemas.runs import (
    MAX_PENDING_COMMANDS,
    MAX_STEPS_PER_RUN,
    CommandAck,
    CommandRequest,
    LoggedStep,
    RunCreate,
    RunFinish,
    RunInfo,
    RunStatus,
)
from app.services.event_stream import Event, EventStream
from app.services.run_health import HealthRules
from app.services.run_store import (
    RUN_SCHEMA_VERSION,
    STORED_EVENTS,
    RunStore,
    StoredRun,
)

logger = logging.getLogger(__name__)

# Alerts a run may publish before further ones are suppressed.
MAX_ALERTS_PER_RUN = 200
# Steps that keep per-layer health details in the replay history; older steps
# keep only the global numbers, which trend charts need.
DETAILED_HEALTH_STEPS = 200
CANCELLED_MESSAGE = "Stopped from NeuroVision."

# A run with no events or heartbeats for this long is marked failed.
IDLE_TIMEOUT_S = 300.0
STALE_MESSAGE = (
    "No events arrived for 5 minutes, so this run was closed. "
    "The script may have stopped without calling run.finish()."
)
FAILED_MESSAGE = "The training script reported a failure."
RESTARTED_MESSAGE = (
    "The NeuroVision backend restarted while this run was active, so it was "
    "closed. The steps logged before the restart are kept."
)
# A running run is saved at most this often; finishing saves it at once.
SAVE_INTERVAL_S = 5.0


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class ExternalRun(EventStream):
    def __init__(
        self,
        request: RunCreate,
        token: str,
        now: float,
        store: RunStore | None = None,
    ) -> None:
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
        # None for a run reloaded from disk: tokens are never stored.
        self._token_digest: bytes | None = _digest(token)
        self._store = store
        self._saved_at = now
        # Serializes ingest, commands, finish, and expiry so checks and
        # publishing agree.
        self.ingest_lock = threading.Lock()
        self._activation_index: int | None = None
        self._visual_index: int | None = None
        self._health_indices: deque[int] = deque()
        # Two-way control (ADR 0008).
        self.controls = list(dict.fromkeys(request.controls))
        self.paused = False
        self.learning_rate: float | None = None
        self._commands: OrderedDict[int, RunCommand] = OrderedDict()
        self._next_command_id = 1
        # Health alerts.
        self._rules = HealthRules()
        self.alerts = 0
        self._next_alert_id = 1
        self.publish(
            TrainingStarted(
                **self._base,
                event_type="training.started",
                model_id=self.model_id or self.run_id,
                total_epochs=self.total_epochs,
            ).model_dump(mode="json")
        )
        self._persist(now, force=True)

    @classmethod
    def restore(cls, stored: StoredRun, store: RunStore, now: float) -> "ExternalRun":
        """A read-only run from disk; one left running is marked failed."""
        # No model_id: imported models live in memory, so after a restart the
        # link would point at a model that no longer exists.
        request = RunCreate(
            name=stored.name,
            total_epochs=stored.total_epochs,
            controls=stored.controls,
        )
        run = cls(request, secrets.token_urlsafe(32), stored.created_at)
        run.run_id = stored.run_id
        run._token_digest = None
        run._events = list(stored.events)
        run.status = stored.status
        run.epoch, run.step = stored.epoch, stored.step
        run.steps_logged = stored.steps_logged
        run.last_event_at = stored.last_event_at
        run.learning_rate = stored.learning_rate
        run.alerts = stored.alerts
        steps = [e for e in stored.events if e["event_type"] == "training.step"]
        run._last_metrics = steps[-1]["metrics"] if steps else {}
        run._store = store
        if run.active:
            with run.ingest_lock:
                run._fail(RESTARTED_MESSAGE, now)
        return run

    def snapshot(self) -> StoredRun:
        with self._lock:
            events = [e for e in self._events if e["event_type"] in STORED_EVENTS]
        return StoredRun(
            schema_version=RUN_SCHEMA_VERSION,
            run_id=self.run_id,
            name=self.name,
            model_id=self.model_id,
            status=self.status,
            total_epochs=self.total_epochs,
            epoch=self.epoch,
            step=self.step,
            steps_logged=self.steps_logged,
            created_at=self.created_at,
            last_event_at=self.last_event_at,
            controls=self.controls,
            learning_rate=self.learning_rate,
            alerts=self.alerts,
            events=events,
        )

    def _persist(self, now: float, force: bool = False) -> None:
        """Save to the store, at most every SAVE_INTERVAL_S unless forced."""
        if self._store is None:
            return
        if not force and now - self._saved_at < SAVE_INTERVAL_S:
            return
        self._saved_at = now
        try:
            self._store.save(self.snapshot())
        except OSError:
            # A full or read-only disk must not stop the live run.
            logger.exception("Could not save run %s", self.run_id)

    def save(self) -> None:
        with self.ingest_lock:
            self._persist(self.last_event_at, force=True)

    def metric_table(self) -> tuple[list[str], list[list[str]]]:
        """Logged metrics by step: the CSV header and rows."""
        with self._lock:
            steps = [e for e in self._events if e["event_type"] == "training.step"]
        names = sorted({name for e in steps for name in e["metrics"]})
        rows = [
            [str(e["step"]), str(e["epoch"])]
            + [repr(e["metrics"][n]) if n in e["metrics"] else "" for n in names]
            for e in steps
        ]
        return ["step", "epoch", *names], rows

    @property
    def _base(self) -> dict[str, str]:
        return {"schema_version": "1.0", "job_id": self.run_id}

    @property
    def active(self) -> bool:
        return self.status == "running"

    @property
    def read_only(self) -> bool:
        """Reloaded from disk: no script can write to it any more."""
        return self._token_digest is None

    def token_matches(self, token: str) -> bool:
        if self._token_digest is None:
            return False
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
            controls=self.controls,
            paused=self.paused,
            learning_rate=self.learning_rate,
            pending_commands=len(self.pending_commands()),
            alerts=self.alerts,
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
                        health=logged.health,
                        system=logged.system,
                        visuals=logged.visuals,
                    ).model_dump(mode="json")
                )
                self.step, self.epoch = logged.step, logged.epoch
                self._last_metrics = logged.metrics
                if (
                    logged.health is not None
                    and logged.health.learning_rate is not None
                ):
                    self.learning_rate = logged.health.learning_rate
                for finding in self._rules.observe(
                    logged.metrics, logged.health, logged.system
                ):
                    self._alert(
                        logged.step,
                        finding.kind,
                        finding.severity,
                        finding.message,
                        finding.advice,
                        finding.layer,
                    )
            self.steps_logged += len(steps)
            self.last_event_at = now
            self._persist(now)
            return len(steps)

    # Commands -------------------------------------------------------------

    def issue(self, request: CommandRequest, now: float) -> RunCommand:
        """Queue a command from the dashboard for the script to apply."""
        with self.ingest_lock:
            self._require_running()
            if request.kind not in self.controls:
                allowed = ", ".join(self.controls) or "none"
                raise ApiError(
                    409,
                    "conflict",
                    f"This run does not accept '{request.kind}' commands (it accepts: "
                    f"{allowed}). Enable them in the script with "
                    "start_run(..., control=True, optimizer=..., on_checkpoint=...).",
                )
            if request.kind == "set_lr" and request.value is None:
                raise invalid("body.value", "Give the new learning rate in 'value'.")
            if request.kind != "set_lr" and request.value is not None:
                raise invalid("body.value", "Only set_lr takes a value.")
            if len(self.pending_commands()) >= MAX_PENDING_COMMANDS:
                raise ApiError(
                    429,
                    "too_many_requests",
                    f"{MAX_PENDING_COMMANDS} commands are already waiting for the "
                    "script. It applies them when it calls run.poll().",
                )
            command = RunCommand(
                command_id=self._next_command_id,
                kind=request.kind,
                value=request.value,
                status="pending",
                created_at=now,
            )
            self._next_command_id += 1
            self._commands[command.command_id] = command
            self._trim_commands()
            self._publish_command(command)
            return command

    def pending_commands(self) -> list[RunCommand]:
        return [c for c in self._commands.values() if c.status == "pending"]

    def acknowledge(self, command_id: int, ack: CommandAck, now: float) -> RunCommand:
        """Record the script's answer to a command."""
        with self.ingest_lock:
            command = self._commands.get(command_id)
            if command is None:
                raise not_found(f"Command {command_id} was not found on this run.")
            if command.status != "pending":
                raise ApiError(
                    409,
                    "conflict",
                    f"Command {command_id} is already {command.status}.",
                )
            if ack.paused is not None:
                self.paused = ack.paused
            if ack.learning_rate is not None:
                self.learning_rate = ack.learning_rate
            updated = command.model_copy(
                update={
                    "status": ack.status,
                    "message": ack.message,
                    "resolved_at": now,
                }
            )
            self._commands[command_id] = updated
            # An answer from the script shows it is alive.
            self.last_event_at = now
            self._publish_command(updated)
            return updated

    def _publish_command(self, command: RunCommand) -> None:
        self.publish(
            RunCommandEvent(
                **self._base,
                event_type="run.command",
                command=command,
                paused=self.paused,
                learning_rate=self.learning_rate,
            ).model_dump(mode="json")
        )

    def _expire_commands(self, now: float, reason: str) -> None:
        for command in self.pending_commands():
            expired = command.model_copy(
                update={"status": "expired", "message": reason, "resolved_at": now}
            )
            self._commands[command.command_id] = expired
            self._publish_command(expired)

    def _trim_commands(self) -> None:
        """Keep every pending command and at most 50 resolved ones."""
        resolved = [cid for cid, c in self._commands.items() if c.status != "pending"]
        for cid in resolved[: max(0, len(resolved) - 50)]:
            del self._commands[cid]

    # Alerts ---------------------------------------------------------------

    def _alert(
        self,
        step: int,
        kind: str,
        severity: str,
        message: str,
        advice: str,
        layer: str | None,
    ) -> None:
        if self.alerts > MAX_ALERTS_PER_RUN:
            return
        if self.alerts == MAX_ALERTS_PER_RUN:
            kind, severity, layer = "alerts_suppressed", "info", None
            message = f"This run raised {MAX_ALERTS_PER_RUN} alerts; further ones are not shown."
            advice = "Fix the reported problems, or start a new run."
        alert = RunAlert.model_validate(
            {
                "alert_id": self._next_alert_id,
                "kind": kind,
                "severity": severity,
                "step": step,
                "layer": layer,
                "message": message[:300],
                "advice": advice[:300],
            }
        )
        self._next_alert_id += 1
        self.alerts += 1
        self.publish(
            RunAlertEvent(**self._base, event_type="run.alert", alert=alert).model_dump(
                mode="json"
            )
        )

    def finish(self, request: RunFinish, now: float) -> None:
        with self.ingest_lock:
            self._require_running()
            self.last_event_at = now
            self._expire_commands(now, "The run ended before the script applied it.")
            if request.status == "cancelled":
                self.status = "cancelled"
                self.paused = False
                self.publish(
                    TrainingCancelled(
                        **self._base,
                        event_type="training.cancelled",
                        step=self.step or 0,
                    ).model_dump(mode="json")
                )
            elif request.status == "completed":
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
                # Saves the run itself.
                self._fail(request.message or FAILED_MESSAGE, now)
                return
            self._persist(now, force=True)

    def expire_if_idle(self, now: float) -> None:
        with self.ingest_lock:
            if self.active and now - self.last_event_at > IDLE_TIMEOUT_S:
                self._fail(STALE_MESSAGE, now)

    def _fail(self, message: str, now: float) -> None:
        self._expire_commands(now, "The run ended before the script applied it.")
        self.status = "failed"
        self.paused = False
        self.publish(
            TrainingFailed(
                **self._base, event_type="training.failed", message=message
            ).model_dump(mode="json")
        )
        self._persist(now, force=True)

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
        if event.get("visuals"):
            # Like activations, keep visuals only on the latest step with them.
            if self._visual_index is not None:
                older = self._events[self._visual_index]
                self._events[self._visual_index] = {**older, "visuals": {}}
            self._visual_index = len(self._events)
        health = event.get("health")
        if health and (health.get("parameters") or health.get("activations")):
            # Per-layer details only on the latest DETAILED_HEALTH_STEPS steps.
            self._health_indices.append(len(self._events))
            if len(self._health_indices) > DETAILED_HEALTH_STEPS:
                index = self._health_indices.popleft()
                older = self._events[index]
                self._events[index] = {
                    **older,
                    "health": {**older["health"], "parameters": {}, "activations": {}},
                }
        self._events.append(event)


class ExternalRunManager:
    def __init__(
        self,
        max_active: int = 8,
        retained: int = 32,
        clock: Callable[[], float] = time.time,
        store: RunStore | None = None,
    ) -> None:
        self._max_active = max_active
        self._retained = retained
        self._clock = clock
        self._store = store
        self._runs: OrderedDict[str, ExternalRun] = OrderedDict()
        self._lock = threading.Lock()
        if store is not None:
            self._reload(store)

    def _reload(self, store: RunStore) -> None:
        """Bring back saved runs, oldest first, within the retention limit."""
        now = self._clock()
        for stored in store.load_all():
            try:
                run = ExternalRun.restore(stored, store, now)
            except ValueError:
                # E.g. a hand-edited name that no longer validates.
                logger.warning("Skipped unreadable saved run %s", stored.run_id)
                continue
            self._runs[run.run_id] = run
        self._trim()

    def flush(self) -> None:
        """Save active runs now, e.g. at shutdown."""
        with self._lock:
            runs = [run for run in self._runs.values() if run.active]
        for run in runs:
            run.save()

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
            run = ExternalRun(request, token, self._clock(), self._store)
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
            if self._store is not None:
                self._store.delete(run_id)
