"""Training jobs: run off the event loop, stream bounded events, cancel safely.

Jobs run in a small thread pool so long training never blocks the ASGI event
loop. Each job keeps its event history (at most epochs + 2 events), so a
WebSocket that connects late, or reconnects, replays it and then follows live.
"""

import asyncio
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

from app.core.errors import ApiError, invalid
from app.ml.datasets import Dataset, DatasetError, make_dataset
from app.ml.trainer import EpochReport, TrainingDiverged, train
from app.schemas.events import (
    TrainingCancelled,
    TrainingCompleted,
    TrainingFailed,
    TrainingStarted,
    TrainingStep,
)
from app.schemas.training import JobStatus, TrainingConfig, TrainingJobInfo
from app.services.model_repository import ModelRecord

logger = logging.getLogger("neurovision.training")

TERMINAL_EVENTS = {"training.completed", "training.cancelled", "training.failed"}
Event = dict[str, Any]
Subscriber = tuple[asyncio.AbstractEventLoop, "asyncio.Queue[Event]"]


class TrainingJob:
    def __init__(self, model_id: str, config: TrainingConfig) -> None:
        self.job_id = f"job_{uuid4().hex[:12]}"
        self.model_id = model_id
        self.config = config
        self.status: JobStatus = "running"
        self.epoch = 0
        self.cancel_requested = threading.Event()
        self._events: list[Event] = []
        self._subscribers: set[Subscriber] = set()
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.status == "running"

    def info(self) -> TrainingJobInfo:
        return TrainingJobInfo(
            job_id=self.job_id,
            model_id=self.model_id,
            status=self.status,
            epoch=self.epoch,
            total_epochs=self.config.epochs,
            config=self.config,
        )

    def publish(self, event: Event) -> None:
        """Record an event and deliver it to every subscriber (any thread)."""
        with self._lock:
            self._events.append(event)
            subscribers = list(self._subscribers)
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:  # The subscriber's event loop has closed.
                self.unsubscribe(queue)

    def subscribe(self) -> tuple[list[Event], "asyncio.Queue[Event]"]:
        """Call from an event loop: past events plus a queue of future ones."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        with self._lock:
            history = list(self._events)
            self._subscribers.add((asyncio.get_running_loop(), queue))
        return history, queue

    def unsubscribe(self, queue: "asyncio.Queue[Event]") -> None:
        with self._lock:
            self._subscribers = {s for s in self._subscribers if s[1] is not queue}


class TrainingJobManager:
    def __init__(self, max_active: int = 2, retained: int = 64) -> None:
        self._max_active = max_active
        self._retained = retained
        self._executor = ThreadPoolExecutor(
            max_workers=max_active, thread_name_prefix="training"
        )
        self._jobs: OrderedDict[str, TrainingJob] = OrderedDict()
        self._lock = threading.Lock()

    def start(self, record: ModelRecord, config: TrainingConfig) -> TrainingJob:
        known = {"input", *record.network.node_ids}
        for i, node_id in enumerate(config.capture_layers):
            if node_id not in known:
                raise invalid(f"body.capture_layers.{i}", f"Unknown layer '{node_id}'.")
        try:
            data = make_dataset(
                config.dataset,
                samples=config.samples,
                noise=config.noise,
                input_units=record.config.input_units,
                output_units=record.config.output_units,
                validation_split=config.validation_split,
                seed=config.seed,
            )
        except DatasetError as error:
            raise invalid("body.dataset", str(error)) from error

        with self._lock:
            active = [job for job in self._jobs.values() if job.active]
            if any(job.model_id == record.model_id for job in active):
                raise ApiError(
                    409,
                    "conflict",
                    "This model is already training. Cancel that job first.",
                )
            if len(active) >= self._max_active:
                raise ApiError(
                    429,
                    "too_many_requests",
                    f"The server is already running {self._max_active} training jobs. "
                    "Try again when one finishes.",
                )
            job = TrainingJob(record.model_id, config)
            self._jobs[job.job_id] = job
            self._trim()
        record.last_job_id = job.job_id
        record.metrics = []
        record.gradients = {}
        self._executor.submit(self._run, job, record, data)
        return job

    def get(self, job_id: str) -> TrainingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> TrainingJob | None:
        job = self.get(job_id)
        if job is not None:
            job.cancel_requested.set()
        return job

    def cancel_for_model(self, model_id: str) -> None:
        with self._lock:
            jobs = [
                j for j in self._jobs.values() if j.model_id == model_id and j.active
            ]
        for job in jobs:
            job.cancel_requested.set()

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.cancel_requested.set()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _trim(self) -> None:
        """Forget the oldest finished jobs past the retention limit."""
        finished = [jid for jid, job in self._jobs.items() if not job.active]
        for job_id in finished[: max(0, len(self._jobs) - self._retained)]:
            del self._jobs[job_id]

    def _run(self, job: TrainingJob, record: ModelRecord, data: Dataset) -> None:
        base = {"schema_version": "1.0", "job_id": job.job_id}
        job.publish(
            TrainingStarted(
                **base,
                event_type="training.started",
                model_id=job.model_id,
                total_epochs=job.config.epochs,
            ).model_dump(mode="json")
        )
        last: EpochReport | None = None

        def on_epoch(report: EpochReport) -> None:
            nonlocal last
            last = report
            job.epoch = report.metrics.epoch
            record.metrics.append(report.metrics)
            job.publish(
                TrainingStep(
                    **base,
                    event_type="training.step",
                    epoch=report.metrics.epoch,
                    step=report.step,
                    metrics=_metrics_dict(report),
                    selected_activations=report.activations,
                ).model_dump(mode="json")
            )

        def on_gradients(gradients: dict) -> None:
            with record.lock:
                record.gradients = gradients

        try:
            finished = train(
                record.network,
                data,
                job.config,
                model_lock=record.lock,
                cancel=job.cancel_requested,
                on_epoch=on_epoch,
                on_gradients=on_gradients,
            )
        except Exception as error:
            if isinstance(error, TrainingDiverged):
                message = str(error)
                logger.info("Training job %s diverged", job.job_id)
            else:
                message = "Training failed unexpectedly."
                logger.exception(
                    "Training job %s for model %s failed", job.job_id, job.model_id
                )
            job.status = "failed"
            job.publish(
                TrainingFailed(
                    **base, event_type="training.failed", message=message
                ).model_dump(mode="json")
            )
            return

        step = last.step if last else 0
        if finished:
            job.status = "completed"
            job.publish(
                TrainingCompleted(
                    **base,
                    event_type="training.completed",
                    epoch=job.epoch,
                    step=step,
                    metrics=_metrics_dict(last) if last else {},
                ).model_dump(mode="json")
            )
        else:
            job.status = "cancelled"
            job.publish(
                TrainingCancelled(
                    **base, event_type="training.cancelled", step=step
                ).model_dump(mode="json")
            )


def _metrics_dict(report: EpochReport) -> dict[str, float]:
    m = report.metrics
    return {
        "loss": m.loss,
        "accuracy": m.accuracy,
        "val_loss": m.val_loss,
        "val_accuracy": m.val_accuracy,
    }
