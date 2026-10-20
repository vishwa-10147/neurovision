"""Child-process entry point that runs ONNX models with ONNX Runtime.

This module is the only place ONNX Runtime is imported, and it only runs in
a separate process started by app/services/onnx_runtime.py. A crash, hang,
or runaway allocation inside the runtime therefore cannot take down the API
process; the parent kills and replaces this worker instead.

Results are summarized here, so only bounded data crosses the process
boundary, however large the model's tensors are.
"""

import time
from collections import OrderedDict
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

MAX_CACHED_SESSIONS = 4


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


def worker_main(conn: Connection, max_memory_bytes: int, threads: int) -> None:
    _limit_memory(max_memory_bytes)
    import onnxruntime as ort

    sessions: OrderedDict[str, Any] = OrderedDict()
    conn.send(("ready", ort.__version__))

    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message[0] == "stop":
            return
        _, key, model_bytes, feeds, max_values = message
        try:
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
            arrays = session.run(names, feeds)
            run_ms = (time.perf_counter() - started) * 1000
            summaries = [
                summarize(n, a, max_values) for n, a in zip(names, arrays, strict=True)
            ]
            conn.send(("ok", (summaries, {"load_ms": load_ms, "run_ms": run_ms})))
        except MemoryError:
            conn.send(("error", "The model ran out of memory."))
        except Exception as error:  # Runtime errors are reported, not raised.
            conn.send(("error", str(error)))
