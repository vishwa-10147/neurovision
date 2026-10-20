"""Run imported ONNX models in an isolated worker process, within limits.

The API process never imports ONNX Runtime. One long-lived worker process
(app/ml/onnx_worker.py) holds sessions for recently run models. Each run
has a time limit: if it is exceeded, or the worker crashes, the worker is
killed and a fresh one starts on the next request. Inputs are validated
against the model's declared inputs before anything reaches the worker.
"""

import hashlib
import multiprocessing
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx

from app.core.errors import ApiError, invalid
from app.ml.onnx_worker import worker_main
from app.schemas.inference import (
    MAX_AUTO_CAPTURE,
    MAX_INPUT_ELEMENTS,
    SUPPORTED_INPUT_DTYPES,
    RunRequest,
    RunResponse,
    TensorValues,
)
from app.schemas.training import MAX_TENSOR_VALUES
from app.services.model_repository import ImportedModelRecord

STARTUP_TIMEOUT_S = 60.0
# How often waits check whether the worker is still alive.
POLL_SLICE_S = 0.05
MAX_ERROR_TEXT = 300


class InferenceTimeout(Exception):
    pass


class InferenceCrashed(Exception):
    pass


class InferenceFailed(Exception):
    """ONNX Runtime rejected the model or its inputs; the message is from it."""


class WorkerBusy(Exception):
    pass


class OnnxRuntimeService:
    def __init__(
        self, timeout_s: float, max_memory_bytes: int, threads: int = 2
    ) -> None:
        self.timeout_s = timeout_s
        self._max_memory = max_memory_bytes
        self._threads = threads
        self._lock = threading.Lock()
        self._process: Any = None
        self._conn: Any = None
        self._loaded: set[str] = set()
        self.version: str | None = None

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=worker_main,
            args=(child, self._max_memory, self._threads),
            daemon=True,
            name="onnx-inference",
        )
        process.start()
        child.close()
        self._process, self._conn, self._loaded = process, parent, set()
        if not self._wait(STARTUP_TIMEOUT_S):
            self._kill()
            raise InferenceCrashed("The inference worker did not start.")
        _, self.version = parent.recv()

    def _wait(self, timeout_s: float) -> bool:
        """Wait for a reply, noticing a dead worker at once instead of at the deadline."""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self._conn.poll(min(POLL_SLICE_S, remaining)):
                return True
            if not self._process.is_alive():
                raise InferenceCrashed("The inference worker exited.")

    def _kill(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.join(5)
        if self._conn is not None:
            self._conn.close()
        self._process, self._conn, self._loaded = None, None, set()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(("stop",))
                except OSError:
                    pass
            self._kill()

    def run(
        self, key: str, model_bytes: Callable[[], bytes], feeds: dict[str, np.ndarray]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Summaries of every output, plus timings measured in the worker."""
        # One run at a time; a second caller waits briefly, then gets "busy".
        if not self._lock.acquire(timeout=self.timeout_s):
            raise WorkerBusy
        try:
            started_worker = False
            if self._process is None or not self._process.is_alive():
                self._kill()
                self._start()
                started_worker = True
            payload = None if key in self._loaded else model_bytes()
            for _attempt in range(2):
                self._conn.send(("run", key, payload, feeds, MAX_TENSOR_VALUES))
                if not self._wait(self.timeout_s):
                    self._kill()
                    raise InferenceTimeout
                status, result = self._conn.recv()
                if status == "need_model":  # The worker evicted it from its cache.
                    payload = model_bytes()
                    continue
                if status == "error":
                    raise InferenceFailed(result)
                self._loaded.add(key)
                summaries, timings = result
                return list(summaries), {**timings, "started_worker": started_worker}
            raise InferenceCrashed("The inference worker did not accept the model.")
        except InferenceCrashed:
            self._kill()
            raise
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as error:
            self._kill()
            raise InferenceCrashed(str(error)) from error
        finally:
            self._lock.release()


@dataclass
class PreparedRun:
    feeds: dict[str, np.ndarray]
    capture: list[str]
    notes: list[str]


def prepare_run(record: ImportedModelRecord, request: RunRequest) -> PreparedRun:
    """Validate inputs against the model's declared inputs and build arrays."""
    if record.blockers:
        raise ApiError(
            422,
            "inference_unsupported",
            "This model cannot be run here. " + " ".join(record.blockers),
        )
    expected = {spec.name: spec for spec in record.inputs}
    for name in request.inputs:
        if name not in expected:
            raise invalid(
                f"body.inputs.{name}", f"The model has no input named '{name}'."
            )
    feeds: dict[str, np.ndarray] = {}
    total = 0
    for spec in record.inputs:
        given = request.inputs.get(spec.name)
        field = f"body.inputs.{spec.name}"
        if given is None:
            raise invalid(field, f"Provide a value for input '{spec.name}'.")
        if spec.shape is not None:
            if len(given.shape) != len(spec.shape):
                raise invalid(
                    f"{field}.shape",
                    f"Expected {len(spec.shape)} dimensions, got {len(given.shape)}.",
                )
            for axis, (want, got) in enumerate(
                zip(spec.shape, given.shape, strict=True)
            ):
                if isinstance(want, int) and want != got:
                    raise invalid(
                        f"{field}.shape.{axis}",
                        f"Dimension {axis} must be {want}, got {got}.",
                    )
        count = int(np.prod(given.shape)) if given.shape else 1
        total += count
        if total > MAX_INPUT_ELEMENTS:
            raise invalid(
                f"{field}.shape",
                f"Inputs may have at most {MAX_INPUT_ELEMENTS:,} values in total.",
            )
        dtype = np.dtype(SUPPORTED_INPUT_DTYPES[spec.dtype])
        if given.values is not None:
            if len(given.values) != count:
                raise invalid(
                    f"{field}.values",
                    f"Expected {count:,} values for shape {given.shape}, got {len(given.values):,}.",
                )
            array = np.asarray(given.values, dtype=np.float64)
            if dtype.kind in "iub":
                if not np.all(np.equal(np.mod(array, 1), 0)):
                    raise invalid(
                        f"{field}.values",
                        f"Input '{spec.name}' is {spec.dtype}; values must be whole numbers.",
                    )
                if dtype.kind != "b":
                    info = np.iinfo(dtype)
                    if array.min() < info.min or array.max() > info.max:
                        raise invalid(
                            f"{field}.values",
                            f"Values must fit {spec.dtype} ({info.min} to {info.max}).",
                        )
            array = array.astype(dtype).reshape(given.shape)
        else:
            array = _fill(given.fill, given.shape, dtype, given.seed)
        feeds[spec.name] = array

    capture, notes = _capture_list(record, request.capture)
    return PreparedRun(feeds=feeds, capture=capture, notes=notes)


def _fill(kind: str, shape: list[int], dtype: np.dtype, seed: int) -> np.ndarray:
    if kind == "zeros":
        return np.zeros(shape, dtype=dtype)
    if kind == "ones":
        return np.ones(shape, dtype=dtype)
    if kind == "range":
        count = int(np.prod(shape)) if shape else 1
        return np.linspace(-1, 1, count).reshape(shape).astype(dtype)
    rng = np.random.default_rng(seed)
    if dtype.kind == "f":
        return rng.standard_normal(shape).astype(dtype)
    if dtype.kind == "b":
        return rng.integers(0, 2, shape).astype(dtype)
    return rng.integers(0, 10, shape).astype(dtype)


def _capture_list(
    record: ImportedModelRecord, requested: list[str] | None
) -> tuple[list[str], list[str]]:
    """Tensor names of operator outputs to return besides the graph outputs."""
    by_node: dict[str, list[str]] = {}
    for tensor, node_id in record.producers.items():
        if node_id.startswith("node_"):
            by_node.setdefault(node_id, []).append(tensor)
    notes: list[str] = []
    if requested is None:
        if record.operator_count > MAX_AUTO_CAPTURE:
            notes.append(
                f"The model has {record.operator_count:,} operators, so intermediate values are not "
                f"captured automatically above {MAX_AUTO_CAPTURE}. Request specific operators with `capture`."
            )
            return [], notes
        requested = sorted(by_node, key=lambda n: int(n.removeprefix("node_")))
    tensors: list[str] = []
    for i, node_id in enumerate(requested):
        if node_id not in by_node:
            raise invalid(
                f"body.capture.{i}", f"'{node_id}' is not an operator in this model."
            )
        tensors.extend(by_node[node_id])
    return tensors, notes


def variant_key(record: ImportedModelRecord, capture: list[str]) -> str:
    digest = hashlib.sha256("\n".join(capture).encode()).hexdigest()[:16]
    return f"{record.model_id}:{digest}"


def build_variant(record: ImportedModelRecord, capture: list[str]) -> bytes:
    """The model with captured intermediate tensors added as graph outputs."""
    model = onnx.load_model_from_string(record.model_bytes)
    existing = {o.name for o in model.graph.output}
    model.graph.output.extend(
        onnx.ValueInfoProto(name=name) for name in capture if name not in existing
    )
    return model.SerializeToString()


def execute(
    service: OnnxRuntimeService, record: ImportedModelRecord, request: RunRequest
) -> RunResponse:
    prepared = prepare_run(record, request)
    try:
        results, timings = service.run(
            variant_key(record, prepared.capture),
            lambda: build_variant(record, prepared.capture),
            prepared.feeds,
        )
    except InferenceTimeout as error:
        raise ApiError(
            504,
            "inference_timeout",
            f"The model did not finish within {service.timeout_s:g} seconds and was stopped.",
        ) from error
    except WorkerBusy as error:
        raise ApiError(
            429, "too_many_requests", "Another model is running. Try again in a moment."
        ) from error
    except InferenceCrashed as error:
        raise ApiError(
            500,
            "inference_crashed",
            "The inference worker stopped unexpectedly; it will restart on the next run.",
        ) from error
    except InferenceFailed as error:
        message = (
            str(error).strip().splitlines()[0]
            if str(error).strip()
            else "unknown error"
        )
        raise ApiError(
            422,
            "inference_failed",
            f"ONNX Runtime could not run this model: {message[:MAX_ERROR_TEXT]}",
        ) from error
    notes = list(prepared.notes)
    if timings["load_ms"]:
        notes.append(
            f"Loading the model into ONNX Runtime took {timings['load_ms']:.0f} ms (cached for later runs)."
        )
    if timings["started_worker"]:
        notes.append("This run also started the inference worker process.")

    graph_outputs = [o for o in record.graph.nodes if o.op_type == "Output"]
    output_names = {n.input_tensors[0] for n in graph_outputs if n.input_tensors}
    values = [
        TensorValues(node_id=record.producers.get(r["name"]), **r) for r in results
    ]
    return RunResponse(
        model_id=record.model_id,
        runtime=f"onnxruntime {service.version}",
        # Model execution only, measured inside the worker.
        elapsed_ms=round(timings["run_ms"], 3),
        outputs=[v for v in values if v.name in output_names],
        intermediates=[v for v in values if v.name not in output_names],
        notes=notes,
    )
