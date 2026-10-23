"""Check each ONNX export against PyTorch, in the backend's ONNX Runtime.

    backend/.venv/Scripts/python bench/check_onnx.py

Runs every bench/exports/<model>.onnx on its saved real-photo input, reports
the largest difference from PyTorch's logits, whether the top-1 class
agrees, and the median CPU latency. Adds them to bench/results/vision_models.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
EXPORTS = HERE / "exports"
RESULTS = HERE / "results" / "vision_models.json"


def main() -> None:
    data = json.loads(RESULTS.read_text())
    for row in data["models"]:
        name = row["model"]
        path = EXPORTS / f"{name}.onnx"
        if not path.exists():
            continue
        x = np.load(EXPORTS / f"{name}.input.npy")
        expected = np.load(EXPORTS / f"{name}.torch.npy")
        graph = onnx.load(str(path), load_external_data=False).graph
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {"input": x})
        times = []
        for _ in range(8):
            start = time.perf_counter()
            session.run(None, {"input": x})
            times.append(time.perf_counter() - start)
        row["onnx_operators"] = len(graph.node)
        row["onnx_max_abs_diff"] = float(np.abs(got - expected).max())
        row["onnx_top1_matches"] = bool(got.argmax() == expected.argmax())
        row["ort_cpu_latency_ms"] = round(1000 * float(np.median(times[2:])), 1)
        print(
            f"{name:20} ops {row['onnx_operators']:5}  max diff {row['onnx_max_abs_diff']:.2e}  "
            f"top-1 {'same' if row['onnx_top1_matches'] else 'DIFFERENT'}  "
            f"ORT CPU {row['ort_cpu_latency_ms']} ms"
        )
    RESULTS.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
