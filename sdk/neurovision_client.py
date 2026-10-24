"""Stream metrics, activations and training health from your own training
script to NeuroVision, and optionally let NeuroVision control the run.

Copy this single file next to your script; it needs only the standard library
(and PyTorch for ActivationProbe and HealthMonitor, which you already have).

    import neurovision_client as nv

    run = nv.start_run("my experiment", total_epochs=EPOCHS)
    for epoch in range(1, EPOCHS + 1):
        ...  # train
        run.log(epoch=epoch, metrics={"loss": loss, "accuracy": acc})
    run.finish()

Training health (gradient, weight, update and activation statistics; the
server turns them into alerts):

    monitor = nv.HealthMonitor(model, optimizer=optimizer, probe_batch=x_val[:64])
    ...
    optimizer.step()
    run.log(epoch=epoch, metrics=..., health=monitor.collect())

Two-way control from the dashboard (pause, resume, stop, checkpoint, change
the learning rate), off unless you opt in:

    run = nv.start_run("x", total_epochs=EPOCHS, control=True,
                       optimizer=optimizer, on_checkpoint=save)
    for batch in loader:
        if not run.poll():   # applies commands; blocks while paused
            break            # stop was requested

Logging never slows training noticeably and never raises into your loop:
events are queued and sent in batches by a background thread; when the
server is slow or unreachable, the oldest queued events are dropped and a
warning is logged once.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = [
    "ActivationProbe",
    "HealthMonitor",
    "Run",
    "SystemMonitor",
    "export_onnx",
    "image_grid",
    "matrix",
    "scatter",
    "start_run",
]

# Limits enforced by the server (backend/app/schemas/events.py, runs.py).
MAX_METRICS = 16
MAX_ACTIVATION_LAYERS = 8
MAX_VALUES_PER_LAYER = 256
MAX_EVENTS_PER_REQUEST = 32
MAX_HEALTH_LAYERS = 32
MAX_NAME_LENGTH = 256
MAX_MESSAGE_LENGTH = 300
MAX_LEARNING_RATE = 10.0
MAX_GPUS = 8
MAX_VISUALS = 6
MAX_SCATTER_POINTS = 512
MAX_MATRIX_SIDE = 64
# Requests stay under the server's 2 MB body limit.
MAX_BATCH_BYTES = 1_500_000

# 127.0.0.1, not localhost: on Windows, "localhost" tries IPv6 first and
# costs about 2 s per request against a server listening on IPv4 only.
DEFAULT_URL = "http://127.0.0.1:8000"
logger = logging.getLogger("neurovision_client")

# (method, url, body or None, headers) -> (status, decoded JSON body or None).
# A bytes body is sent as-is (an ONNX file); anything else as JSON.
Transport = Callable[[str, str, Any, dict[str, str]], tuple[int, Any]]


def urllib_transport(timeout: float) -> Transport:
    def send(
        method: str, url: str, body: Any, headers: dict[str, str]
    ) -> tuple[int, Any]:
        raw = isinstance(body, bytes)
        data = (
            body if raw else None if body is None else json.dumps(body).encode("utf-8")
        )
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header(
            "Content-Type", "application/octet-stream" if raw else "application/json"
        )
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, _json(response.read())
        except urllib.error.HTTPError as error:
            return error.code, _json(error.read())

    return send


def _json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _error_message(body: Any) -> str:
    try:
        return str(body["error"]["message"])
    except (KeyError, TypeError):
        return "no details"


def start_run(
    name: str,
    *,
    total_epochs: int,
    url: str | None = None,
    model_id: str | None = None,
    model: Any = None,
    example_input: Any = None,
    control: bool = False,
    optimizer: Any = None,
    on_checkpoint: Callable[[], Any] | None = None,
    track_system: bool = True,
    flush_interval: float = 0.5,
    control_interval: float = 1.0,
    max_queue: int = 512,
    timeout: float = 5.0,
    transport: Transport | None = None,
) -> Run:
    """Create a run on the NeuroVision backend and return a logger for it.

    `url` defaults to $NEUROVISION_URL, then http://127.0.0.1:8000. If the
    backend cannot be reached, a warning is logged and the returned run
    silently ignores everything, so your training still runs.

    `control=True` lets the dashboard pause, resume and stop the run;
    `optimizer` adds changing the learning rate, and `on_checkpoint` (called
    with no arguments; may return a short description such as a file name)
    adds saving a checkpoint. Commands take effect when you call run.poll().
    Anyone who can reach the backend can send them, so keep it on localhost
    or a trusted network.

    `model` and `example_input` (a PyTorch module and one input batch) link
    the run to the network's graph: the model is exported to ONNX here and
    imported into NeuroVision, and Live Runs shows that graph in 2D and 3D,
    colored by the activations and gradients each step reports. Use module
    names (e.g. "net.1") as ActivationProbe labels so they map onto the
    graph; HealthMonitor statistics use them already. If the export or the
    upload fails (e.g. a file over the server's upload limit), a warning is
    logged and the run is created without a graph.

    `track_system` (on by default) sends timing (elapsed, step time, ETA,
    throughput) with every step, plus CPU, RAM and GPU use when psutil and
    nvidia-ml-py are installed (both optional); see SystemMonitor.
    """
    base = (url or os.environ.get("NEUROVISION_URL") or DEFAULT_URL).rstrip("/")
    send = transport or urllib_transport(timeout)
    controls: list[str] = []
    if control:
        controls = ["pause", "resume", "stop"]
        if optimizer is not None:
            controls.append("set_lr")
        if on_checkpoint is not None:
            controls.append("checkpoint")
    elif optimizer is not None or on_checkpoint is not None:
        logger.info(
            "NeuroVision: pass control=True to start_run to let the dashboard "
            "use the optimizer or checkpoint function."
        )
    if model_id is None and model is not None and example_input is not None:
        model_id = _link_model(
            transport or urllib_transport(max(timeout, 120.0)),
            base,
            name,
            model,
            example_input,
        )
    body: dict[str, Any] = {"name": name, "total_epochs": total_epochs}
    if model_id is not None:
        body["model_id"] = model_id
    if controls:
        body["controls"] = controls
    try:
        status, payload = send("POST", f"{base}/api/v1/runs", body, {})
    except Exception as error:  # Network failures must not stop training.
        logger.warning("NeuroVision: cannot reach %s (%s); not logging.", base, error)
        return Run.disabled(name)
    if status != 201 or not isinstance(payload, dict):
        logger.warning(
            "NeuroVision: could not start a run (HTTP %s: %s); not logging.",
            status,
            _error_message(payload),
        )
        return Run.disabled(name)
    run = Run(
        name=name,
        run_id=str(payload["run"]["run_id"]),
        token=str(payload["write_token"]),
        base_url=base,
        transport=send,
        flush_interval=flush_interval,
        max_queue=max_queue,
        controls=controls,
        optimizer=optimizer,
        on_checkpoint=on_checkpoint,
        control_interval=control_interval,
        total_epochs=total_epochs,
        system=SystemMonitor(device=_device_of(model)) if track_system else None,
    )
    logger.info("NeuroVision: logging run %s to %s", run.run_id, base)
    return run


def scatter(
    x: Any,
    y: Any,
    labels: Any = None,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
) -> dict[str, Any]:
    """A 2D scatter plot for run.log(visuals=...): at most 512 points.

    `labels` (optional) are integer classes 0–99, one per point, for colors.
    Longer inputs keep their first 512 points.
    """
    xs, ys = _values(x)[:MAX_SCATTER_POINTS], _values(y)[:MAX_SCATTER_POINTS]
    n = min(len(xs), len(ys))
    visual: dict[str, Any] = {"kind": "scatter", "x": xs[:n], "y": ys[:n]}
    if labels is not None:
        visual["labels"] = [max(0, min(99, int(v))) for v in _values(labels)[:n]]
        if len(visual["labels"]) != n:
            raise ValueError("labels must have one entry per point")
    if x_label:
        visual["x_label"] = x_label[:40]
    if y_label:
        visual["y_label"] = y_label[:40]
    return visual


def matrix(
    values: Any,
    *,
    scale: str = "diverging",
    row_label: str | None = None,
    col_label: str | None = None,
) -> dict[str, Any]:
    """A 2D matrix for run.log(visuals=...): heatmaps, attention, images.

    `values` is 2D (a tensor, array or nested lists), at most 64 × 64; larger
    ones are cropped to the top-left 64 × 64. `scale="sequential"` draws low
    to high (probabilities, pixels); "diverging" centers on zero.
    """
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    rows = [
        list(map(float, row))[:MAX_MATRIX_SIDE]
        for row in list(values)[:MAX_MATRIX_SIDE]
    ]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError(
            "matrix() needs a non-empty 2D array with rows of equal length"
        )
    visual: dict[str, Any] = {
        "kind": "matrix",
        "rows": len(rows),
        "cols": len(rows[0]),
        "values": [v for row in rows for v in row],
        "scale": "sequential" if scale == "sequential" else "diverging",
    }
    if row_label:
        visual["row_label"] = row_label[:40]
    if col_label:
        visual["col_label"] = col_label[:40]
    return visual


def image_grid(images: Any, columns: int = 4, padding: int = 1) -> list[list[float]]:
    """Tile small 2D images (e.g. generated 8 × 8 digits) into one matrix.

    `images` has shape [n, height, width]; tiles are separated by `padding`
    cells of 0. The result fits matrix() when the mosaic is at most 64 × 64.
    """
    if hasattr(images, "detach"):
        images = images.detach().float().cpu().tolist()
    elif hasattr(images, "tolist"):
        images = images.tolist()
    images = list(images)
    height, width = len(images[0]), len(images[0][0])
    rows = (len(images) + columns - 1) // columns
    step_y, step_x = height + padding, width + padding
    grid = [
        [0.0] * (columns * step_x - padding) for _ in range(rows * step_y - padding)
    ]
    for k, image in enumerate(images):
        top, left = (k // columns) * step_y, (k % columns) * step_x
        for r in range(height):
            for c in range(width):
                grid[top + r][left + c] = float(image[r][c])
    return grid


def _device_of(model: Any) -> str | None:
    """The device of a PyTorch model's first parameter, e.g. "cpu" or "cuda:0"."""
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return None


def export_onnx(model: Any, example_input: Any) -> bytes:
    """The model as ONNX bytes (opset 17, dynamic batch), for NeuroVision.

    Exported in eval mode under torch.no_grad(); the model's mode is restored.
    """
    import io
    import warnings

    import torch

    buffer = io.BytesIO()
    was_training = model.training
    model.eval()
    try:
        # The TorchScript exporter (dynamo=False) names nodes by module path
        # ("/layer1/layer1.0/relu/Relu"), which Live Runs uses to place each
        # layer's values on the graph; it warns that it is deprecated.
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                model,
                (example_input,),
                buffer,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
    finally:
        model.train(was_training)
    return buffer.getvalue()


def _short_reason(error: Exception) -> str:
    """The error's type and first line: exporter errors can embed a whole traced graph."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    first = next(
        (line for line in lines if not line.startswith(("%", "graph(", "Torch IR"))), ""
    )
    return (
        f"{type(error).__name__}: {first}"[:MAX_MESSAGE_LENGTH]
        if first
        else type(error).__name__
    )


def _link_model(
    send: Transport, base: str, name: str, model: Any, example_input: Any
) -> str | None:
    """Export and import the model; its id, or None (with a warning)."""
    try:
        data = export_onnx(model, example_input)
    except Exception as error:
        logger.warning(
            "NeuroVision: could not export the model to ONNX (%s); no graph.",
            _short_reason(error),
        )
        return None
    filename = (
        "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:200] or "model"
    )
    query = urllib.parse.urlencode({"filename": f"{filename}.onnx"})
    try:
        status, payload = send("POST", f"{base}/api/v1/import/onnx?{query}", data, {})
    except Exception as error:
        logger.warning("NeuroVision: could not upload the model (%s); no graph.", error)
        return None
    if status != 201 or not isinstance(payload, dict) or "model_id" not in payload:
        logger.warning(
            "NeuroVision: the model was not imported (HTTP %s: %s; %.1f MB); no graph.",
            status,
            _error_message(payload),
            len(data) / 2**20,
        )
        return None
    logger.info("NeuroVision: linked the run to model %s", payload["model_id"])
    return str(payload["model_id"])


class Run:
    """Logs steps to one external run. Thread-safe; methods never raise."""

    # Send an empty batch when idle this long, so the server keeps the run open.
    HEARTBEAT_S = 60.0
    # How often poll() re-checks for commands while the run is paused.
    PAUSE_CHECK_S = 0.2

    def __init__(
        self,
        *,
        name: str,
        run_id: str | None,
        token: str | None,
        base_url: str,
        transport: Transport | None,
        flush_interval: float = 0.5,
        max_queue: int = 512,
        controls: Iterable[str] = (),
        optimizer: Any = None,
        on_checkpoint: Callable[[], Any] | None = None,
        control_interval: float = 1.0,
        total_epochs: int | None = None,
        system: SystemMonitor | None = None,
    ) -> None:
        self.name = name
        self.run_id = run_id
        self._token = token
        self._base = base_url
        self._send = transport
        self._flush_interval = flush_interval
        self._queue: deque[dict[str, Any]] = deque(maxlen=max_queue)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._stopped = threading.Event()
        self._enabled = run_id is not None and transport is not None
        self._finished = False
        self._last_step = -1
        self._last_sent = time.monotonic()
        self._warned: set[str] = set()
        self.dropped = 0
        self.sent = 0
        # Two-way control: commands arrive on the background thread and are
        # applied in the training thread by poll().
        self.controls = list(controls)
        self.optimizer = optimizer
        self.on_checkpoint = on_checkpoint
        self._control_interval = control_interval
        self._last_fetch = 0.0
        self._inbox: deque[dict[str, Any]] = deque()
        self._seen: set[int] = set()
        self._acks: deque[tuple[int, dict[str, Any]]] = deque()
        self.paused = False
        self.stop_requested = False
        self.commands_applied = 0
        # Timing and resource use for each logged step.
        self.total_epochs = total_epochs
        self._system = system
        self._started = time.monotonic()
        self._last_log: float | None = None
        self._progress: deque[tuple[float, float]] = deque(maxlen=20)
        self._thread: threading.Thread | None = None
        if self._enabled:
            self._thread = threading.Thread(
                target=self._loop, name="neurovision-client", daemon=True
            )
            self._thread.start()
            atexit.register(self._at_exit)

    @classmethod
    def disabled(cls, name: str) -> Run:
        return cls(name=name, run_id=None, token=None, base_url="", transport=None)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def controllable(self) -> bool:
        return bool(self.controls)

    # Public API ---------------------------------------------------------

    def log(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, Any],
        step: int | None = None,
        activations: Mapping[str, Any] | None = None,
        health: Mapping[str, Any] | None = None,
        samples: int | None = None,
        progress: float | None = None,
        visuals: Mapping[str, Any] | None = None,
    ) -> None:
        """Queue one step. `step` defaults to one more than the last step.

        `samples` (optional) is how many training examples this step
        covers since the previous log() call, for a throughput figure.
        `visuals` (optional) maps names to scatter() or matrix() results,
        e.g. a 2D latent space or attention weights; at most 6 per step.
        `progress` (optional) is the fractional number of epochs done, e.g.
        epoch - 1 + batch / batches, so the ETA is right when you log several
        times per epoch; without it the ETA follows `epoch`.

        Metric values and activations may be floats, lists, NumPy arrays, or
        tensors (they are copied to the CPU here). Values that the server
        would reject (too many metrics or layers) are dropped with a one-time
        warning instead of raising. Non-finite metrics are not sent, but
        their names are reported so the dashboard can alert on them.
        `health` is what HealthMonitor.collect() returns.
        """
        if not self._enabled or self._finished:
            return
        try:
            event = self._build(epoch, step, metrics, activations or {}, health)
            if event is not None and visuals:
                cleaned_visuals = self._clean_visuals(visuals)
                if cleaned_visuals:
                    event["visuals"] = cleaned_visuals
            if event is not None:
                system = self._system_stats(
                    epoch if progress is None else progress, samples
                )
                if system:
                    event["system"] = system
        except Exception as error:
            self._warn_once("log", f"skipped a step it could not read ({error}).")
            return
        if event is None:
            return
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
                self._warn_once(
                    "backpressure",
                    "the server is not keeping up; dropping the oldest queued steps.",
                )
            self._queue.append(event)
            self._idle.clear()
            if len(self._queue) >= MAX_EVENTS_PER_REQUEST:
                self._wake.set()

    def poll(self) -> bool:
        """Apply commands from the dashboard; call it often, e.g. once per batch.

        Blocks while the run is paused (still answering resume, stop,
        set_lr and checkpoint). Returns False once a stop was requested, so
        you can save and leave your loop; True otherwise. Cheap when there
        are no commands: it only checks an in-memory queue.
        """
        if not self.controls:
            return not self.stop_requested
        self._apply_received()
        while (
            self.paused
            and not self.stop_requested
            and self._enabled
            and not self._finished
        ):
            self._last_fetch = 0.0  # Fetch again right away.
            self._wake.set()
            time.sleep(self.PAUSE_CHECK_S)
            self._apply_received()
        return not self.stop_requested

    @property
    def should_stop(self) -> bool:
        return self.stop_requested

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until queued steps are sent (or dropped). True if drained."""
        if not self._enabled:
            return True
        self._wake.set()
        return self._idle.wait(timeout)

    def finish(
        self,
        status: str = "completed",
        message: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Send what is queued, then mark the run completed or failed.

        A run stopped from the dashboard finishes as "cancelled" unless
        another status is given. Waits up to `timeout` seconds; steps still
        unsent then are dropped (and counted), so nothing arrives after the
        run is finished.
        """
        if not self._enabled or self._finished:
            return
        self._finished = True
        if self.stop_requested and status == "completed":
            status = "cancelled"
        self.flush(timeout)
        self._stopped.set()
        self._wake.set()
        if self._thread is not None:
            # Let an in-flight request complete before the run is closed.
            self._thread.join(timeout)
        with self._lock:
            if self._queue:
                self.dropped += len(self._queue)
                self._queue.clear()
                self._warn_once("unsent", "finish() timed out; dropped unsent steps.")
        self._send_acks()
        body: dict[str, Any] = {"status": status}
        if message:
            body["message"] = message[:500]
        self._post(f"/api/v1/runs/{self.run_id}/finish", body)
        if self.dropped:
            logger.warning("NeuroVision: %d steps were dropped.", self.dropped)

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.finish()
        else:
            # The type only: exception messages can contain paths or data.
            self.finish("failed", f"The training script raised {exc_type.__name__}.")

    # Building steps -----------------------------------------------------

    def _build(
        self,
        epoch: int,
        step: int | None,
        metrics: Mapping[str, Any],
        activations: Mapping[str, Any],
        health: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        with self._lock:
            step = self._last_step + 1 if step is None else int(step)
            if step <= self._last_step:
                self._warn_once("step", "skipped a step whose number did not increase.")
                return None
            self._last_step = step
        clean: dict[str, float] = {}
        nonfinite: list[str] = []
        for key, value in metrics.items():
            number = _scalar(value)
            if not math.isfinite(number):
                self._warn_once(
                    "nan", f"metric '{key}' was not finite; reported by name."
                )
                if len(nonfinite) < MAX_METRICS:
                    nonfinite.append(str(key)[:MAX_NAME_LENGTH])
                continue
            if len(clean) == MAX_METRICS:
                self._warn_once(
                    "metrics", f"kept only the first {MAX_METRICS} metrics."
                )
                break
            clean[str(key)] = number
        layers: dict[str, list[float]] = {}
        for key, value in activations.items():
            if len(layers) == MAX_ACTIVATION_LAYERS:
                self._warn_once(
                    "layers", f"kept only the first {MAX_ACTIVATION_LAYERS} layers."
                )
                break
            values = _values(value)
            if len(values) > MAX_VALUES_PER_LAYER:
                values = values[:MAX_VALUES_PER_LAYER]
            if not values or not all(math.isfinite(v) for v in values):
                self._warn_once(
                    "nan-activations", f"skipped empty or non-finite layer '{key}'."
                )
                continue
            layers[str(key)[:MAX_NAME_LENGTH]] = values
        event: dict[str, Any] = {
            "epoch": max(0, int(epoch)),
            "step": step,
            "metrics": clean,
            "selected_activations": layers,
        }
        cleaned = _clean_health(health, nonfinite)
        if cleaned is not None:
            event["health"] = cleaned
        return event

    def _clean_visuals(self, visuals: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, visual in visuals.items():
            if len(out) == MAX_VISUALS:
                self._warn_once(
                    "visuals", f"kept only the first {MAX_VISUALS} visuals."
                )
                break
            if not isinstance(visual, Mapping) or visual.get("kind") not in (
                "scatter",
                "matrix",
            ):
                self._warn_once(
                    "visual-kind",
                    f"skipped visual '{name}': use nv.scatter() or nv.matrix().",
                )
                continue
            values = [v for key in ("x", "y", "values") for v in visual.get(key, [])]
            if not all(math.isfinite(v) for v in values):
                self._warn_once(
                    "visual-nan",
                    f"skipped visual '{name}' with NaN or infinite values.",
                )
                continue
            out[str(name)[:MAX_NAME_LENGTH]] = dict(visual)
        return out

    def _system_stats(self, epoch: float, samples: int | None) -> dict[str, Any]:
        """Timing (always) and resource use (when available) for this step."""
        if self._system is None:
            return {}
        now = time.monotonic()
        stats: dict[str, Any] = {"elapsed_s": now - self._started}
        if self._last_log is not None:
            step_time = now - self._last_log
            stats["step_time_s"] = step_time
            if samples and step_time > 0:
                stats["samples_per_s"] = samples / step_time
        self._last_log = now
        # ETA from the recent rate of epochs per second.
        self._progress.append((now, float(epoch)))
        if self.total_epochs and len(self._progress) >= 2:
            (t0, e0), (t1, e1) = self._progress[0], self._progress[-1]
            if e1 > e0 and t1 > t0:
                rate = (e1 - e0) / (t1 - t0)
                stats["eta_s"] = max(0.0, (self.total_epochs - e1) / rate)
        try:
            stats.update(self._system.sample())
        except Exception as error:
            self._warn_once("system", f"could not read system stats ({error}).")
        return _clean_system(stats)

    # Background thread --------------------------------------------------

    def _loop(self) -> None:
        while not self._stopped.is_set():
            self._wake.wait(self._flush_interval)
            self._wake.clear()
            self._drain()
            if self.controls and self._enabled:
                now = time.monotonic()
                if now - self._last_fetch >= self._control_interval:
                    self._fetch_commands()
                self._send_acks()
            if (
                not self._stopped.is_set()
                and time.monotonic() - self._last_sent > self.HEARTBEAT_S
            ):
                self._post_events([])

    def _drain(self) -> None:
        while True:
            with self._lock:
                batch: list[dict[str, Any]] = []
                size = 0
                # Up to 32 steps, and under the server's body limit when
                # steps carry large visuals.
                while self._queue and len(batch) < MAX_EVENTS_PER_REQUEST:
                    event_size = len(json.dumps(self._queue[0]))
                    if batch and size + event_size > MAX_BATCH_BYTES:
                        break
                    batch.append(self._queue.popleft())
                    size += event_size
                if not batch:
                    self._idle.set()
                    return
            if self._enabled:
                self._post_events(batch)
            else:
                self.dropped += len(batch)

    def _post_events(self, batch: list[dict[str, Any]]) -> None:
        status, payload = self._post(
            f"/api/v1/runs/{self.run_id}/events", {"events": batch}
        )
        if status == 202:
            self.sent += len(batch)
            if isinstance(payload, dict):
                self._receive(payload.get("commands"))
            return
        self.dropped += len(batch)
        if status in (401, 403, 404, 409):
            # The run is gone, finished, full, or not ours: stop sending.
            self._enabled = False
            self._warn_once(
                "stopped",
                f"stopped logging (HTTP {status}: {_error_message(payload)}).",
            )
        elif status is not None:
            self._warn_once(
                f"http-{status}",
                f"the server rejected steps (HTTP {status}: {_error_message(payload)}).",
            )

    # Commands -----------------------------------------------------------

    def _fetch_commands(self) -> None:
        self._last_fetch = time.monotonic()
        status, payload = self._request(
            "GET", f"/api/v1/runs/{self.run_id}/commands", None
        )
        if status == 200 and isinstance(payload, dict):
            self._receive(payload.get("commands"))
        elif status in (401, 403, 404):
            self._enabled = False
            self._warn_once("stopped", f"stopped (HTTP {status}).")

    def _receive(self, commands: Any) -> None:
        if not isinstance(commands, list):
            return
        with self._lock:
            for command in commands:
                try:
                    command_id = int(command["command_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if command_id not in self._seen:
                    self._seen.add(command_id)
                    self._inbox.append(command)

    def _apply_received(self) -> None:
        while True:
            with self._lock:
                if not self._inbox:
                    return
                command = self._inbox.popleft()
            self._apply(command)

    def _apply(self, command: Mapping[str, Any]) -> None:
        """Run one command in the training thread and queue the script's answer."""
        kind = command.get("kind")
        status, message = "applied", None
        if kind not in self.controls:
            status, message = "rejected", f"This script does not accept '{kind}'."
        elif kind == "pause":
            message = (
                "Already paused."
                if self.paused
                else f"Paused after step {max(self._last_step, 0)}."
            )
            self.paused = True
        elif kind == "resume":
            if self.paused:
                self.paused = False
                message = "Resumed."
            else:
                status, message = "rejected", "The run is not paused."
        elif kind == "stop":
            self.stop_requested = True
            self.paused = False
            message = "Stopping: run.poll() now returns False."
        elif kind == "checkpoint":
            status, message = self._checkpoint()
        elif kind == "set_lr":
            status, message = self._set_learning_rate(command.get("value"))
        if status == "applied":
            self.commands_applied += 1
        ack: dict[str, Any] = {"status": status, "paused": self.paused}
        if message:
            ack["message"] = message[:MAX_MESSAGE_LENGTH]
        rate = self._learning_rate()
        if rate is not None:
            ack["learning_rate"] = rate
        with self._lock:
            self._acks.append((int(command["command_id"]), ack))
        self._wake.set()
        logger.info(
            "NeuroVision run '%s': %s %s (%s)", self.name, kind, status, message
        )

    def _checkpoint(self) -> tuple[str, str]:
        if self.on_checkpoint is None:
            return "rejected", "No on_checkpoint function was given to start_run."
        try:
            result = self.on_checkpoint()
        except Exception as error:
            # The type only: messages can contain paths or data.
            return "rejected", f"on_checkpoint raised {type(error).__name__}."
        detail = f": {result}" if isinstance(result, str) and result else ""
        return "applied", f"Checkpoint saved{detail}."

    def _set_learning_rate(self, value: Any) -> tuple[str, str]:
        if self.optimizer is None:
            return "rejected", "No optimizer was given to start_run."
        try:
            rate = float(value)
        except (TypeError, ValueError):
            return "rejected", "The learning rate was not a number."
        if not (math.isfinite(rate) and 0 < rate <= MAX_LEARNING_RATE):
            return (
                "rejected",
                f"The learning rate must be > 0 and <= {MAX_LEARNING_RATE:g}.",
            )
        try:
            for group in self.optimizer.param_groups:
                group["lr"] = rate
        except Exception as error:
            return "rejected", f"Setting lr raised {type(error).__name__}."
        return "applied", f"Learning rate set to {rate:g}. A scheduler may override it."

    def _learning_rate(self) -> float | None:
        try:
            rate = float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return None
        return rate if math.isfinite(rate) and rate >= 0 else None

    def _send_acks(self) -> None:
        while True:
            with self._lock:
                if not self._acks:
                    return
                command_id, ack = self._acks.popleft()
            status, payload = self._post(
                f"/api/v1/runs/{self.run_id}/commands/{command_id}/ack", ack
            )
            if status is None:
                # Network trouble: retry later, in order.
                with self._lock:
                    self._acks.appendleft((command_id, ack))
                return
            if status not in (200, 404, 409):
                self._warn_once(
                    f"ack-{status}",
                    f"could not report a command result (HTTP {status}: {_error_message(payload)}).",
                )

    # Transport ----------------------------------------------------------

    def _post(self, path: str, body: Any) -> tuple[int | None, Any]:
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: Any) -> tuple[int | None, Any]:
        assert self._send is not None
        if method == "POST":
            self._last_sent = time.monotonic()
        try:
            return self._send(
                method,
                f"{self._base}{path}",
                body,
                {"Authorization": f"Bearer {self._token}"},
            )
        except Exception as error:
            self._warn_once("network", f"could not reach the server ({error}).")
            return None, None

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning("NeuroVision run '%s': %s", self.name, message)

    def _at_exit(self) -> None:
        if not self._finished:
            self.finish("failed", "The script exited without calling run.finish().")


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _values(value: Any) -> list[float]:
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().flatten()[:MAX_VALUES_PER_LAYER].float().cpu().tolist()
    elif hasattr(value, "ravel"):  # numpy.ndarray
        value = value.ravel()[:MAX_VALUES_PER_LAYER].tolist()
    return [float(v) for v in _flatten(value)]


def _flatten(value: Any) -> Iterable[Any]:
    if isinstance(value, list | tuple):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


# System statistics ----------------------------------------------------------


class SystemMonitor:
    """CPU, RAM and GPU use of the machine and this process.

    Uses psutil for CPU/RAM and NVML (pip install nvidia-ml-py) for GPU
    utilization, memory, temperature and power, when they are installed;
    without them only timing is reported. If PyTorch is already imported and
    CUDA is initialized, its allocator's memory use is reported too.
    """

    def __init__(self, device: str | None = None) -> None:
        # Where the model trains ("cpu", "cuda:0"), when known from its
        # parameters. CUDA being initialized is not enough: in PyTorch 2.14
        # optimizer.step() initializes it even for models on the CPU.
        self.device = device
        try:
            import psutil

            self._psutil: Any = psutil
            self._process = psutil.Process()
            psutil.cpu_percent(None)  # Start the measuring interval.
            self._process.cpu_percent(None)
        except Exception:
            self._psutil = None
        self._nvml: Any = None
        self._handles: list[Any] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            count = min(pynvml.nvmlDeviceGetCount(), MAX_GPUS)
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._nvml = pynvml
        except Exception:
            self._nvml = None

    @property
    def has_gpu_stats(self) -> bool:
        return bool(self._handles)

    def sample(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        if self._psutil is not None:
            memory = self._psutil.virtual_memory()
            stats.update(
                cpu_percent=self._psutil.cpu_percent(None),
                process_cpu_percent=self._process.cpu_percent(None),
                ram_used_bytes=float(memory.total - memory.available),
                ram_total_bytes=float(memory.total),
                process_rss_bytes=float(self._process.memory_info().rss),
            )
        gpus = [self._gpu(index, handle) for index, handle in enumerate(self._handles)]
        if gpus:
            stats["gpus"] = gpus
        if self.device is not None:
            stats["device"] = self.device
        torch = sys.modules.get("torch")
        on_cuda = self.device is None or self.device.startswith("cuda")
        if torch is not None and on_cuda:
            try:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    stats["cuda_allocated_bytes"] = float(torch.cuda.memory_allocated())
                    stats["cuda_reserved_bytes"] = float(torch.cuda.memory_reserved())
                    stats.setdefault("device", f"cuda:{torch.cuda.current_device()}")
            except Exception:
                pass
        return stats

    def _gpu(self, index: int, handle: Any) -> dict[str, Any]:
        nvml = self._nvml
        gpu: dict[str, Any] = {"index": index}
        try:
            name = nvml.nvmlDeviceGetName(handle)
            gpu["name"] = name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            pass
        try:
            gpu["util_percent"] = float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            pass
        try:
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            gpu["mem_used_bytes"] = float(memory.used)
            gpu["mem_total_bytes"] = float(memory.total)
        except Exception:
            pass
        try:
            gpu["temperature_c"] = float(nvml.nvmlDeviceGetTemperature(handle, 0))
        except Exception:
            pass
        try:
            gpu["power_w"] = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass
        return gpu


_SYSTEM_NUMBERS = (
    "elapsed_s",
    "step_time_s",
    "eta_s",
    "samples_per_s",
    "process_cpu_percent",
    "ram_used_bytes",
    "ram_total_bytes",
    "process_rss_bytes",
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
)


def _clean_system(stats: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _SYSTEM_NUMBERS:
        number = _finite(stats.get(key))
        if number is not None:
            out[key] = number
    cpu = _finite(stats.get("cpu_percent"), maximum=100.0)
    if cpu is not None:
        out["cpu_percent"] = cpu
    if isinstance(stats.get("device"), str):
        out["device"] = stats["device"][:64]
    gpus = []
    for gpu in list(stats.get("gpus") or [])[:MAX_GPUS]:
        entry: dict[str, Any] = {"index": max(0, int(gpu.get("index", 0)))}
        if isinstance(gpu.get("name"), str):
            entry["name"] = gpu["name"][:MAX_MESSAGE_LENGTH]
        util = _finite(gpu.get("util_percent"), maximum=100.0)
        if util is not None:
            entry["util_percent"] = util
        for key in ("mem_used_bytes", "mem_total_bytes", "temperature_c", "power_w"):
            number = _finite(gpu.get(key))
            if number is not None:
                entry[key] = number
        gpus.append(entry)
    if gpus:
        out["gpus"] = gpus
    return out


# Health statistics ----------------------------------------------------------


def _finite(
    value: Any, *, minimum: float | None = 0.0, maximum: float | None = None
) -> float | None:
    """A finite float within bounds, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def _clean_health(
    health: Mapping[str, Any] | None, nonfinite_metrics: list[str]
) -> dict[str, Any] | None:
    """Keep only what the server accepts, so one bad number never drops a step."""
    if health is None and not nonfinite_metrics:
        return None
    health = health or {}
    out: dict[str, Any] = {}
    for key in ("grad_norm", "learning_rate"):
        number = _finite(health.get(key))
        if number is not None:
            out[key] = number
    try:
        out["nonfinite_grads"] = max(0, int(health.get("nonfinite_grads", 0)))
    except (TypeError, ValueError):
        out["nonfinite_grads"] = 0
    if nonfinite_metrics:
        out["nonfinite_metrics"] = nonfinite_metrics[:MAX_METRICS]
    parameters: dict[str, Any] = {}
    for name, group in list((health.get("parameters") or {}).items())[
        :MAX_HEALTH_LAYERS
    ]:
        entry: dict[str, Any] = {}
        for key in ("grad_norm", "weight_norm", "update_ratio"):
            number = _finite(group.get(key))
            if number is not None:
                entry[key] = number
        try:
            entry["parameters"] = max(0, int(group.get("parameters", 0)))
        except (TypeError, ValueError):
            entry["parameters"] = 0
        parameters[str(name)[:MAX_NAME_LENGTH]] = entry
    if parameters:
        out["parameters"] = parameters
    activations: dict[str, Any] = {}
    for name, layer in list((health.get("activations") or {}).items())[
        :MAX_HEALTH_LAYERS
    ]:
        mean = _finite(layer.get("mean"), minimum=None)
        std = _finite(layer.get("std"))
        kind = layer.get("kind")
        if (
            mean is None
            or std is None
            or kind not in ("relu", "tanh", "sigmoid", "other")
        ):
            continue
        entry = {"kind": kind, "mean": mean, "std": std}
        for key in ("dead_fraction", "saturated_fraction"):
            number = _finite(layer.get(key), maximum=1.0)
            if number is not None:
                entry[key] = number
        activations[str(name)[:MAX_NAME_LENGTH]] = entry
    if activations:
        out["activations"] = activations
    return out


_ACTIVATION_KINDS = {
    "ReLU": "relu",
    "ReLU6": "relu",
    "Tanh": "tanh",
    "Sigmoid": "sigmoid",
    "LeakyReLU": "other",
    "ELU": "other",
    "GELU": "other",
    "SiLU": "other",
}


def _group_parameters(
    named: list[tuple[str, Any]], max_groups: int
) -> OrderedDict[str, list[Any]]:
    """Group parameters by the module that owns them, merging when there are many."""

    def owner(name: str, depth: int | None) -> str:
        parts = name.split(".")[:-1]
        if depth is not None:
            parts = parts[:depth]
        return ".".join(parts) or "(root)"

    groups: OrderedDict[str, list[Any]] = OrderedDict()
    for depth in (None, 1):
        groups = OrderedDict()
        for name, parameter in named:
            groups.setdefault(owner(name, depth), []).append(parameter)
        if len(groups) <= max_groups:
            return groups
    items = list(groups.items())
    merged = [p for _, params in items[max_groups - 1 :] for p in params]
    return OrderedDict([*items[: max_groups - 1], ("other", merged)])


class HealthMonitor:
    """Measure training-health statistics for NeuroVision's alerts.

    Call collect() right after optimizer.step() (before the next
    zero_grad()), as often as you log, e.g. once per epoch or every N
    batches, and pass the result to run.log(health=...). It measures:

    - per parameter group (module that owns parameters): gradient norm,
      weight norm, and the update ratio ‖Δw‖/‖w‖ (keeps one copy of the
      parameters; track_updates=False turns it off). With an `optimizer`,
      its steps are counted and the ratio is the change since the previous
      collect() divided by the number of steps, i.e. the average per step,
      which is what the alert thresholds assume; without one it is the
      whole change since the previous collect();
    - globally: total gradient norm, the optimizer's learning rate, and the
      number of NaN/∞ gradient values;
    - with a probe_batch, for nn.ReLU/ReLU6/LeakyReLU/ELU/GELU/SiLU/Tanh/
      Sigmoid modules: output mean and std, the fraction of dead units
      (<= 0 for every probe example; ReLU/ReLU6), and the fraction of
      saturated outputs (tanh, sigmoid). Functional activations (F.relu)
      are not seen. A module used several times reports its last use.

    Activation hooks are attached only during collect(), which runs the
    probe batch in eval mode under torch.no_grad() and restores the mode.
    """

    def __init__(
        self,
        model: Any,
        *,
        optimizer: Any = None,
        probe_batch: Any = None,
        track_updates: bool = True,
        max_layers: int = MAX_HEALTH_LAYERS,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.probe_batch = probe_batch
        self.track_updates = track_updates
        max_layers = max(1, min(max_layers, MAX_HEALTH_LAYERS))
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self._groups = _group_parameters(trainable, max_layers)
        self._activation_modules = [
            (
                name or type(module).__name__,
                module,
                _ACTIVATION_KINDS[type(module).__name__],
            )
            for name, module in model.named_modules()
            if type(module).__name__ in _ACTIVATION_KINDS
        ][:max_layers]
        self._previous: dict[int, Any] = {}
        # Optimizer steps since the previous collect(); None when not counted.
        self._steps: int | None = None
        if optimizer is not None and hasattr(optimizer, "register_step_post_hook"):
            self._steps = 0
            optimizer.register_step_post_hook(self._count_step)

    def _count_step(self, *_: Any) -> None:
        if self._steps is not None:
            self._steps += 1

    def collect(self) -> dict[str, Any]:
        import torch

        health: dict[str, Any] = {"parameters": {}, "nonfinite_grads": 0}
        steps = max(1, self._steps) if self._steps is not None else 1
        if self._steps is not None:
            self._steps = 0
        total_sq = 0.0
        any_grad = False
        with torch.no_grad():
            for name, params in self._groups.items():
                grad_sq = weight_sq = update_sq = 0.0
                has_grad = False
                has_update = self.track_updates
                bad = 0
                count = 0
                for p in params:
                    count += p.numel()
                    data = p.detach()
                    weight_sq += float(data.float().pow(2).sum())
                    if p.grad is not None:
                        grad = p.grad.detach()
                        nonfinite = int((~torch.isfinite(grad)).sum())
                        bad += nonfinite
                        if nonfinite == 0:
                            grad_sq += float(grad.float().pow(2).sum())
                            has_grad = True
                    if self.track_updates:
                        previous = self._previous.get(id(p))
                        if previous is None or previous.shape != data.shape:
                            self._previous[id(p)] = data.clone()
                            has_update = False
                        else:
                            update_sq += float((data - previous).float().pow(2).sum())
                            previous.copy_(data)
                health["nonfinite_grads"] += bad
                weight_norm = math.sqrt(weight_sq)
                entry: dict[str, Any] = {
                    "parameters": count,
                    "weight_norm": weight_norm,
                }
                if has_grad and bad == 0:
                    entry["grad_norm"] = math.sqrt(grad_sq)
                    total_sq += grad_sq
                    any_grad = True
                if has_update and weight_norm > 0:
                    entry["update_ratio"] = math.sqrt(update_sq) / weight_norm / steps
                health["parameters"][name] = entry
        if any_grad:
            health["grad_norm"] = math.sqrt(total_sq)
        if self.optimizer is not None:
            try:
                health["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
            except Exception:
                pass
        if self.probe_batch is not None and self._activation_modules:
            health["activations"] = self._activations()
        return health

    def _activations(self) -> dict[str, Any]:
        import torch

        outputs: dict[str, Any] = {}

        def hook(name: str) -> Callable[..., None]:
            def record(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, list | tuple) else output
                if isinstance(tensor, torch.Tensor):
                    outputs[name] = tensor.detach()

            return record

        handles = [
            module.register_forward_hook(hook(name))
            for name, module, _ in self._activation_modules
        ]
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                self.model(self.probe_batch)
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)
        stats: dict[str, Any] = {}
        for name, _, kind in self._activation_modules:
            if name in outputs:
                stats[name] = _activation_stats(outputs[name], kind)
        return stats


def _activation_stats(output: Any, kind: str) -> dict[str, Any]:
    values = output.float()
    entry: dict[str, Any] = {
        "kind": kind,
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)) if values.numel() > 1 else 0.0,
    }
    if kind == "relu":
        # A unit is a feature (dim 1: channels for conv outputs) across the batch.
        units = (
            values.transpose(0, 1).reshape(values.shape[1], -1)
            if values.dim() >= 2
            else values.reshape(1, -1)
        )
        entry["dead_fraction"] = float((units.max(dim=1).values <= 0).float().mean())
    elif kind == "tanh":
        entry["saturated_fraction"] = float((values.abs() > 0.99).float().mean())
    elif kind == "sigmoid":
        entry["saturated_fraction"] = float(
            ((values < 0.01) | (values > 0.99)).float().mean()
        )
    return entry


class ActivationProbe:
    """Measure chosen layers' outputs on one fixed probe batch.

    Pass the modules whose outputs you want, e.g. the activation modules
    (nn.ReLU) for post-activation values:

        probe = ActivationProbe(model, {"hidden_1": model.act1}, x_val[:1])
        run.log(epoch=epoch, metrics=..., activations=probe.capture())

    Hooks are attached only for the duration of capture(), so training
    forward passes pay nothing. capture() runs in eval mode under
    torch.no_grad() and restores the model's previous mode. Values are the
    first example's output, flattened, keeping the first 256 values.
    """

    @classmethod
    def by_name(cls, model: Any, names: list[str], probe_batch: Any) -> ActivationProbe:
        """Probe modules by their names in model.named_modules(), e.g. "net.1".

        Module names as labels let Live Runs place the values on the model's
        graph when the run is linked to it (start_run(model=...)).
        """
        modules = dict(model.named_modules())
        missing = [n for n in names if n not in modules]
        if missing:
            raise ValueError(f"No such modules: {', '.join(missing)}")
        return cls(model, {n: modules[n] for n in names}, probe_batch)

    def __init__(self, model: Any, layers: Mapping[str, Any], probe_batch: Any):
        if not layers:
            raise ValueError("Choose at least one layer to probe.")
        if len(layers) > MAX_ACTIVATION_LAYERS:
            raise ValueError(f"Probe at most {MAX_ACTIVATION_LAYERS} layers.")
        self.model = model
        self.layers = dict(layers)
        self.probe_batch = probe_batch

    def capture(self) -> dict[str, list[float]]:
        import torch

        captured: dict[str, list[float]] = {}

        def hook(name: str) -> Callable[..., None]:
            def record(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, list | tuple) else output
                if isinstance(tensor, torch.Tensor):
                    captured[name] = _values(tensor[0] if tensor.dim() > 0 else tensor)

            return record

        handles = [m.register_forward_hook(hook(n)) for n, m in self.layers.items()]
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                self.model(self.probe_batch)
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)
        return captured
