"""Stream metrics and activations from your own training script to NeuroVision.

Copy this single file next to your script; it needs only the standard library
(and PyTorch for ActivationProbe, which you already have).

    import neurovision_client as nv

    run = nv.start_run("my experiment", total_epochs=EPOCHS)
    for epoch in range(1, EPOCHS + 1):
        ...  # train
        run.log(epoch=epoch, metrics={"loss": loss, "accuracy": acc})
    run.finish()

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
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = ["ActivationProbe", "Run", "start_run"]

# Limits enforced by the server (backend/app/schemas/events.py, runs.py).
MAX_METRICS = 16
MAX_ACTIVATION_LAYERS = 8
MAX_VALUES_PER_LAYER = 256
MAX_EVENTS_PER_REQUEST = 32

# 127.0.0.1, not localhost: on Windows, "localhost" tries IPv6 first and
# costs about 2 s per request against a server listening on IPv4 only.
DEFAULT_URL = "http://127.0.0.1:8000"
logger = logging.getLogger("neurovision_client")

# (method, url, body or None, headers) -> (status, decoded JSON body or None)
Transport = Callable[[str, str, Any, dict[str, str]], tuple[int, Any]]


def urllib_transport(timeout: float) -> Transport:
    def send(
        method: str, url: str, body: Any, headers: dict[str, str]
    ) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
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
    flush_interval: float = 0.5,
    max_queue: int = 512,
    timeout: float = 5.0,
    transport: Transport | None = None,
) -> Run:
    """Create a run on the NeuroVision backend and return a logger for it.

    `url` defaults to $NEUROVISION_URL, then http://127.0.0.1:8000. If the
    backend cannot be reached, a warning is logged and the returned run
    silently ignores everything, so your training still runs.
    """
    base = (url or os.environ.get("NEUROVISION_URL") or DEFAULT_URL).rstrip("/")
    send = transport or urllib_transport(timeout)
    body: dict[str, Any] = {"name": name, "total_epochs": total_epochs}
    if model_id is not None:
        body["model_id"] = model_id
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
    )
    logger.info("NeuroVision: logging run %s to %s", run.run_id, base)
    return run


class Run:
    """Logs steps to one external run. Thread-safe; methods never raise."""

    # Send an empty batch when idle this long, so the server keeps the run open.
    HEARTBEAT_S = 60.0

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

    # Public API ---------------------------------------------------------

    def log(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, Any],
        step: int | None = None,
        activations: Mapping[str, Any] | None = None,
    ) -> None:
        """Queue one step. `step` defaults to one more than the last step.

        Metric values and activations may be floats, lists, NumPy arrays, or
        tensors (they are copied to the CPU here). Values that the server
        would reject (non-finite numbers, too many metrics or layers) are
        dropped with a one-time warning instead of raising.
        """
        if not self._enabled or self._finished:
            return
        try:
            event = self._build(epoch, step, metrics, activations or {})
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

        Waits up to `timeout` seconds; steps still unsent then are dropped
        (and counted), so nothing arrives after the run is finished.
        """
        if not self._enabled or self._finished:
            return
        self._finished = True
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

    # Internals ----------------------------------------------------------

    def _build(
        self,
        epoch: int,
        step: int | None,
        metrics: Mapping[str, Any],
        activations: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        with self._lock:
            step = self._last_step + 1 if step is None else int(step)
            if step <= self._last_step:
                self._warn_once("step", "skipped a step whose number did not increase.")
                return None
            self._last_step = step
        clean: dict[str, float] = {}
        for key, value in metrics.items():
            number = _scalar(value)
            if not math.isfinite(number):
                self._warn_once("nan", f"dropped non-finite metric '{key}'.")
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
            layers[str(key)[:256]] = values
        return {
            "epoch": max(0, int(epoch)),
            "step": step,
            "metrics": clean,
            "selected_activations": layers,
        }

    def _loop(self) -> None:
        while not self._stopped.is_set():
            self._wake.wait(self._flush_interval)
            self._wake.clear()
            self._drain()
            if (
                not self._stopped.is_set()
                and time.monotonic() - self._last_sent > self.HEARTBEAT_S
            ):
                self._post_events([])

    def _drain(self) -> None:
        while True:
            with self._lock:
                batch = [
                    self._queue.popleft()
                    for _ in range(min(MAX_EVENTS_PER_REQUEST, len(self._queue)))
                ]
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

    def _post(self, path: str, body: Any) -> tuple[int | None, Any]:
        assert self._send is not None
        self._last_sent = time.monotonic()
        try:
            return self._send(
                "POST",
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
