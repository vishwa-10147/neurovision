"""Benchmark pretrained torchvision models and export them for NeuroVision.

For each model: parameters, GFLOPs, top-5 predictions on two real photos
(the sample images that ship with scikit-learn), latency on the GPU and CPU,
GPU throughput with mixed precision, peak GPU memory, CPU/GPU load while
running, and an ONNX export that NeuroVision's Model Explorer can import.

    bench/.venv-gpu/Scripts/python bench/vision_models.py            # all
    bench/.venv-gpu/Scripts/python bench/vision_models.py resnet50   # one

Pretrained weights are downloaded once by torchvision (into bench/weights).
Results go to bench/results/vision_models.json; ONNX files to bench/exports.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(HERE / "weights"))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import pynvml  # noqa: E402
import torch  # noqa: E402
import torchvision.models as tvm  # noqa: E402
import torchvision.models.video as tvv  # noqa: E402
from sklearn.datasets import load_sample_images  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

RESULTS = HERE / "results"
EXPORTS = HERE / "exports"


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    builder: Any
    weights: Any
    # Input shape without the batch dimension.
    shape: tuple[int, ...]
    task: str = "image classification (ImageNet-1k)"
    batch: int = 64


SPECS = [
    Spec(
        "mobilenet_v3_large",
        "CNN (mobile)",
        tvm.mobilenet_v3_large,
        tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        (3, 224, 224),
    ),
    Spec(
        "efficientnet_b0",
        "CNN (mobile)",
        tvm.efficientnet_b0,
        tvm.EfficientNet_B0_Weights.IMAGENET1K_V1,
        (3, 224, 224),
    ),
    Spec(
        "resnet18",
        "CNN (residual)",
        tvm.resnet18,
        tvm.ResNet18_Weights.IMAGENET1K_V1,
        (3, 224, 224),
    ),
    Spec(
        "resnet50",
        "CNN (residual)",
        tvm.resnet50,
        tvm.ResNet50_Weights.IMAGENET1K_V2,
        (3, 224, 224),
    ),
    Spec(
        "convnext_tiny",
        "CNN (modern)",
        tvm.convnext_tiny,
        tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        (3, 224, 224),
    ),
    Spec(
        "vit_b_16",
        "Vision Transformer",
        tvm.vit_b_16,
        tvm.ViT_B_16_Weights.IMAGENET1K_V1,
        (3, 224, 224),
        batch=32,
    ),
    Spec(
        "r3d_18",
        "3D CNN (video)",
        tvv.r3d_18,
        tvv.R3D_18_Weights.KINETICS400_V1,
        (3, 16, 112, 112),
        task="video action recognition (Kinetics-400)",
        batch=16,
    ),
]


class LoadSampler:
    """Samples CPU and GPU load in the background while a block runs."""

    def __init__(self, gpu: Any, interval: float = 0.1) -> None:
        self.gpu, self.interval = gpu, interval
        self.cpu: list[float] = []
        self.util: list[float] = []
        self.power: list[float] = []
        self.temp: list[float] = []
        self._stop = threading.Event()

    def __enter__(self) -> LoadSampler:
        psutil.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.cpu.append(psutil.cpu_percent(None))
            if self.gpu is not None:
                self.util.append(pynvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu)
                self.power.append(pynvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000)
                self.temp.append(pynvml.nvmlDeviceGetTemperature(self.gpu, 0))

    def summary(self) -> dict[str, float | None]:
        def mean(v: list[float]) -> float | None:
            return round(float(np.mean(v)), 1) if v else None

        return {
            "cpu_percent": mean(self.cpu),
            "gpu_util_percent": mean(self.util),
            "gpu_power_w": mean(self.power),
            "gpu_temp_max_c": max(self.temp) if self.temp else None,
        }


def photos(spec: Spec, transform: Any) -> tuple[list[str], torch.Tensor]:
    """The two scikit-learn sample photos, preprocessed for this model."""
    data = load_sample_images()
    names = [Path(f).stem for f in data.filenames]
    images = [torch.from_numpy(img).permute(2, 0, 1) for img in data.images]
    if len(spec.shape) == 4:
        # A video model gets a 16-frame clip: a slow pan across the photo.
        clips = []
        for img in images:
            frames = [img[:, :, i * 8 : i * 8 + 520] for i in range(spec.shape[1])]
            clips.append(transform(torch.stack(frames)))
        return names, torch.stack(clips)
    return names, torch.stack([transform(img) for img in images])


def timed(fn: Any, repeats: int, sync: bool) -> list[float]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def bench(spec: Spec, gpu: Any) -> dict[str, Any]:
    print(f"\n== {spec.name} ==", flush=True)
    t0 = time.perf_counter()
    model = spec.builder(weights=spec.weights).eval()
    load_s = time.perf_counter() - t0
    categories = spec.weights.meta["categories"]
    params = sum(p.numel() for p in model.parameters())
    one = torch.randn(1, *spec.shape)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(one)
    gflops = counter.get_total_flops() / 2e9  # Multiply-adds, as papers report.

    names, batch = photos(spec, spec.weights.transforms())
    result: dict[str, Any] = {
        "model": spec.name,
        "family": spec.family,
        "task": spec.task,
        "weights": str(spec.weights),
        "input": [1, *spec.shape],
        "params_m": round(params / 1e6, 2),
        "gmacs": round(gflops, 2),
        "reported_top1": spec.weights.meta["_metrics"]
        .get("ImageNet-1K", spec.weights.meta["_metrics"].get("Kinetics-400", {}))
        .get("acc@1"),
        "load_s": round(load_s, 2),
    }

    # Predictions on the real photos (on the CPU, fp32).
    with torch.no_grad():
        probs = model(batch).softmax(-1)
    result["predictions"] = {
        name: [
            {"label": categories[i], "p": round(float(p), 4)}
            for p, i in zip(*probs[k].topk(5), strict=True)
        ]
        for k, name in enumerate(names)
    }
    for name in names:
        top = result["predictions"][name][0]
        print(f"  {name}: {top['label']} ({top['p']:.1%})")

    # CPU latency, batch 1.
    with torch.no_grad():
        cpu_times = timed(lambda: model(one), 12, sync=False)[2:]
    result["cpu_latency_ms"] = round(1000 * float(np.median(cpu_times)), 1)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        model = model.to(dev)
        x1 = one.to(dev)
        with torch.no_grad():
            for _ in range(5):
                model(x1)
            torch.cuda.synchronize()
            gpu_times = timed(lambda: model(x1), 30, sync=True)
        result["gpu_latency_ms"] = round(1000 * float(np.median(gpu_times)), 2)

        xb = torch.randn(spec.batch, *spec.shape, device=dev)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model(xb)
            torch.cuda.synchronize()
            with LoadSampler(gpu) as load:
                start = time.perf_counter()
                runs = 0
                while time.perf_counter() - start < 4.0:
                    model(xb)
                    runs += 1
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
        result["gpu_batch"] = spec.batch
        result["gpu_throughput_per_s"] = round(runs * spec.batch / elapsed, 1)
        result["gpu_peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        result["load_during_throughput"] = load.summary()
        print(
            f"  GPU {result['gpu_latency_ms']} ms (b1) | "
            f"{result['gpu_throughput_per_s']:.0f}/s (b{spec.batch}, fp16) | "
            f"CPU {result['cpu_latency_ms']} ms | load {load.summary()}"
        )
        model = model.cpu()
        del xb, x1
        torch.cuda.empty_cache()

    # ONNX export for the Model Explorer (classic exporter, opset 17).
    EXPORTS.mkdir(exist_ok=True)
    path = EXPORTS / f"{spec.name}.onnx"
    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        (one,),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    result["onnx_export_s"] = round(time.perf_counter() - t0, 2)
    result["onnx_mib"] = round(path.stat().st_size / 2**20, 1)
    # A reference input/output pair, to check ONNX Runtime against PyTorch.
    np.save(EXPORTS / f"{spec.name}.input.npy", batch[:1].numpy())
    with torch.no_grad():
        np.save(EXPORTS / f"{spec.name}.torch.npy", model(batch[:1]).numpy())
    print(f"  ONNX {result['onnx_mib']} MiB in {result['onnx_export_s']} s")
    return result


def main() -> None:
    wanted = set(sys.argv[1:])
    gpu = None
    try:
        pynvml.nvmlInit()
        gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        pass
    torch.backends.cudnn.benchmark = True
    machine = {
        "cpu": platform.processor(),
        "cpu_threads": psutil.cpu_count(),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    print(machine)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "vision_models.json"
    previous = json.loads(out.read_text()) if out.exists() else {"models": []}
    results = {r["model"]: r for r in previous["models"]}
    for spec in SPECS:
        if wanted and spec.name not in wanted:
            continue
        results[spec.name] = bench(spec, gpu)
        out.write_text(
            json.dumps(
                {
                    "machine": machine,
                    "models": [results[s.name] for s in SPECS if s.name in results],
                },
                indent=2,
            )
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
