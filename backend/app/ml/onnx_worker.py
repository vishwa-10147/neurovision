"""Child-process entry point that runs ONNX models with ONNX Runtime.

This module is the only place ONNX Runtime is imported, and it only runs in
a separate process started by app/services/onnx_runtime.py. A crash, hang,
or runaway allocation inside the runtime therefore cannot take down the API
process; the parent kills and replaces this worker instead.

Results are summarized here, so only bounded data crosses the process
boundary, however large the model's tensors are. The tensors of the last
few runs are kept (within a memory budget) so 2D views of feature maps and
attention can be requested afterwards without running the model again.
"""

import time
from collections import OrderedDict
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from app.ml.tensor_maps import MapRequestError, tensor_grid, tensor_map

MAX_CACHED_SESSIONS = 4
MAX_RETAINED_RUNS = 4
MAX_RETAINED_BYTES = 256 * 1024 * 1024


def _limit_memory(max_bytes: int) -> None:
    """Cap the worker's address space where the OS supports it (Linux/macOS)."""
    try:
        import resource
    except ImportError:  # Windows: no rlimits; see docs/importing-onnx.md.
        return
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except (ValueError, OSError):
        pass


def summarize(name: str, array: np.ndarray, max_values: int) -> dict[str, Any]:
    """Bounded description of one tensor: shape, finite-value stats, a slice."""
    flat = np.asarray(array).reshape(-1)
    numeric = flat.dtype.kind in "biuf"
    values = flat.astype(np.float64) if numeric else np.zeros(0)
    finite = values[np.isfinite(values)] if numeric else values
    window = values[:max_values]
    return {
        "name": name,
        "shape": [int(d) for d in np.shape(array)],
        "dtype": str(flat.dtype),
        "count": int(flat.size),
        "summary": (
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
            }
            if finite.size
            else None
        ),
        "non_finite_count": int(values.size - finite.size) if numeric else 0,
        "values": [float(v) if np.isfinite(v) else None for v in window],
        "truncated": int(flat.size) > len(window),
    }


def as_array(name: str, value: Any) -> tuple[np.ndarray, str | None]:
    """An output as an array, with a note when it had to be converted.

    ONNX-ML models can return non-tensor outputs, e.g. scikit-learn's ZipMap:
    a list of {class: probability} maps, one per row.
    """
    if isinstance(value, np.ndarray):
        return value, None
    if isinstance(value, list) and value and isinstance(value[0], dict):
        keys = sorted(value[0])
        array = np.array([[row[k] for k in keys] for row in value], dtype=np.float32)
        shown = ", ".join(str(k) for k in keys[:10]) + (", …" if len(keys) > 10 else "")
        return array, (
            f"Output '{name}' is a list of maps (ZipMap); shown as a [batch, {len(keys)}] tensor "
            f"with columns ordered by class: {shown}. Export with zipmap=False to get a tensor."
        )
    return np.asarray(
        value
    ), f"Output '{name}' was not a tensor; converted to an array."


class RetainedRuns:
    """Tensors of recent runs (inputs and outputs), for later 2D views."""

    def __init__(self) -> None:
        self._runs: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def add(self, run_id: str, tensors: dict[str, np.ndarray]) -> bool:
        size = sum(a.nbytes for a in tensors.values())
        if size > MAX_RETAINED_BYTES:
            return False
        self._runs[run_id] = tensors
        while len(self._runs) > MAX_RETAINED_RUNS or self._bytes() > MAX_RETAINED_BYTES:
            self._runs.popitem(last=False)
        return True

    def get(self, run_id: str, name: str) -> np.ndarray | None:
        tensors = self._runs.get(run_id)
        return None if tensors is None else tensors.get(name)

    def _bytes(self) -> int:
        return sum(a.nbytes for run in self._runs.values() for a in run.values())


def worker_main(conn: Connection, max_memory_bytes: int, threads: int) -> None:
    _limit_memory(max_memory_bytes)
    import onnxruntime as ort

    sessions: OrderedDict[str, Any] = OrderedDict()
    retained = RetainedRuns()
    conn.send(("ready", ort.__version__))

    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        kind = message[0]
        if kind == "stop":
            return
        try:
            if kind in ("map", "grid"):
                _, run_id, name, fixed, limits = message
                array = retained.get(run_id, name)
                if array is None:
                    conn.send(("gone", None))
                elif kind == "map":
                    conn.send(("ok", tensor_map(array, fixed, limits["max_cells"])))
                else:
                    conn.send(
                        (
                            "ok",
                            tensor_grid(
                                array, fixed, limits["max_items"], limits["thumb_cells"]
                            ),
                        )
                    )
                continue

            _, key, model_bytes, feeds, max_values, run_id = message
            load_ms = 0.0
            session = sessions.get(key)
            if session is None:
                started = time.perf_counter()
                if model_bytes is None:
                    conn.send(("need_model", None))
                    continue
                options = ort.SessionOptions()
                options.intra_op_num_threads = threads
                options.inter_op_num_threads = 1
                # Basic optimizations keep every requested intermediate tensor.
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                )
                options.enable_mem_pattern = False
                # CPU only: never use providers that could reach the network.
                session = ort.InferenceSession(
                    model_bytes, options, providers=["CPUExecutionProvider"]
                )
                sessions[key] = session
                while len(sessions) > MAX_CACHED_SESSIONS:
                    sessions.popitem(last=False)
                load_ms = (time.perf_counter() - started) * 1000
            sessions.move_to_end(key)
            names = [o.name for o in session.get_outputs()]
            started = time.perf_counter()
            raw = session.run(names, feeds)
            run_ms = (time.perf_counter() - started) * 1000
            converted = [as_array(n, v) for n, v in zip(names, raw, strict=True)]
            arrays = [a for a, _ in converted]
            output_notes = [note for _, note in converted if note]
            summaries = [
                summarize(n, a, max_values) for n, a in zip(names, arrays, strict=True)
            ]
            kept = retained.add(
                run_id, {**feeds, **dict(zip(names, arrays, strict=True))}
            )
            timings = {
                "load_ms": load_ms,
                "run_ms": run_ms,
                "retained": kept,
                "notes": output_notes,
            }
            conn.send(("ok", (summaries, timings)))
        except MapRequestError as error:
            conn.send(("bad_request", str(error)))
        except MemoryError:
            conn.send(("error", "The model ran out of memory."))
        except Exception as error:  # Runtime errors are reported, not raised.
            conn.send(("error", str(error)))
