"""Fine-tune a pretrained vision model on the GPU and watch it in NeuroVision.

Real data with no download: scikit-learn's 1,797 handwritten digits (8×8),
upsampled on the GPU to the model's input size. The run streams loss,
accuracy, test accuracy, training health and system use (CPU, RAM, GPU
load/memory/temperature/power, step time, throughput, ETA) to Live Runs,
and can be paused, stopped or given a new learning rate from the dashboard.

    bench/.venv-gpu/Scripts/python bench/train_live.py --model resnet50
    bench/.venv-gpu/Scripts/python bench/train_live.py --model vit_b_16 --epochs 4

Start the backend first (http://127.0.0.1:8001 with the dev launch config).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(HERE / "weights"))
sys.path.insert(0, str(HERE.parent / "sdk"))

import neurovision_client as nv  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.models as tvm  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch import nn  # noqa: E402

MODELS = {
    "mobilenet_v3_large": (
        tvm.mobilenet_v3_large,
        tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    ),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "vit_b_16": (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_V1),
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build(name: str, classes: int) -> nn.Module:
    builder, weights = MODELS[name]
    model = builder(weights=weights)
    # Replace the ImageNet head with a 10-class one.
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif name == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, classes)
    else:
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, classes)
    return model


def to_images(x: torch.Tensor, size: int, augment: bool) -> torch.Tensor:
    """8×8 digits (0–16) to normalized 3-channel images, on the GPU."""
    x = (x / 16.0).view(-1, 1, 8, 8)
    if augment:
        # A random shift of up to one pixel, so the model cannot memorize.
        shift = torch.randint(-1, 2, (2,)).tolist()
        x = torch.roll(x, shifts=shift, dims=(2, 3))
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x.repeat(1, 3, 1, 1) - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, size: int, batch: int
) -> tuple[float, float]:
    model.eval()
    loss = correct = 0.0
    for i in range(0, len(x), batch):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(to_images(x[i : i + batch], size, False))
        loss += F.cross_entropy(
            logits.float(), y[i : i + batch], reduction="sum"
        ).item()
        correct += (logits.argmax(1) == y[i : i + batch]).sum().item()
    model.train()
    return loss / len(x), correct / len(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="resnet50")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument(
        "--no-health",
        action="store_true",
        help="skip HealthMonitor (to measure its cost)",
    )
    parser.add_argument(
        "--bn-momentum",
        type=float,
        default=None,
        help="override BatchNorm momentum (torchvision builds MobileNetV3 with 0.01)",
    )
    parser.add_argument(
        "--log-every", type=int, default=4, help="batches between logged steps"
    )
    parser.add_argument(
        "--url", default=os.environ.get("NEUROVISION_URL", "http://127.0.0.1:8001")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("This benchmark needs a CUDA GPU.")

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.2,
        random_state=0,
        stratify=digits.target,
    )
    x_train, x_test = (
        torch.tensor(a, dtype=torch.float32, device=dev) for a in (x_train, x_test)
    )
    y_train, y_test = (torch.tensor(a, device=dev) for a in (y_train, y_test))

    t0 = time.perf_counter()
    model = build(args.model, 10).to(dev)
    if args.bn_momentum is not None:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = args.bn_momentum
    setup_s = time.perf_counter() - t0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    batches = (len(x_train) + args.batch - 1) // args.batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * batches, pct_start=0.15
    )
    scaler = torch.amp.GradScaler()
    monitor = nv.HealthMonitor(
        model, optimizer=optimizer, probe_batch=to_images(x_test[:16], args.size, False)
    )

    summary: dict[str, object] = {
        "model": args.model,
        "bn_momentum": args.bn_momentum,
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }
    step = 0
    train_start = time.perf_counter()
    with nv.start_run(
        f"{args.model} · digits fine-tune",
        total_epochs=args.epochs,
        url=args.url,
        control=True,
        optimizer=optimizer,
    ) as run:
        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(len(x_train), device=dev)
            seen = correct = 0
            running = 0.0
            for b in range(batches):
                if not run.poll():  # Waits while paused; False after a stop.
                    break
                idx = order[b * args.batch : (b + 1) * args.batch]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(to_images(x_train[idx], args.size, True))
                    loss = F.cross_entropy(logits, y_train[idx])
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler skips steps whose fp16 gradients overflowed and
                # lowers its scale; those steps' gradients are inf by design.
                overflowed = scaler.get_scale() < scale
                if not overflowed:
                    scheduler.step()
                seen += len(idx)
                correct += (logits.argmax(1) == y_train[idx]).sum().item()
                running += loss.item() * len(idx)
                step += 1
                if step % args.log_every == 0 or b == batches - 1:
                    metrics = {"loss": running / seen, "accuracy": correct / seen}
                    if b == batches - 1:
                        test_loss, test_acc = evaluate(
                            model, x_test, y_test, args.size, args.batch * 2
                        )
                        metrics |= {"val_loss": test_loss, "val_accuracy": test_acc}
                    run.log(
                        epoch=epoch,
                        step=step,
                        metrics=metrics,
                        health=None
                        if overflowed or args.no_health
                        else monitor.collect(),
                        samples=args.log_every * args.batch,
                        progress=epoch - 1 + (b + 1) / batches,
                    )
            if run.should_stop:
                break
            print(
                f"epoch {epoch}: loss {running / max(seen, 1):.4f} train {correct / max(seen, 1):.3f} "
                f"test {metrics.get('val_accuracy', float('nan')):.3f}",
                flush=True,
            )
        train_s = time.perf_counter() - train_start

    t0 = time.perf_counter()
    test_loss, test_acc = evaluate(model, x_test, y_test, args.size, args.batch * 2)
    torch.cuda.synchronize()
    test_s = time.perf_counter() - t0
    summary |= {
        "epochs": args.epochs,
        "train_images": len(x_train),
        "test_images": len(x_test),
        "setup_s": round(setup_s, 2),
        "train_s": round(train_s, 1),
        "train_images_per_s": round(args.epochs * len(x_train) / train_s, 1),
        "test_s": round(test_s, 2),
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "stopped": run.should_stop,
    }
    print(json.dumps(summary, indent=2))
    out = HERE / "results" / "training.json"
    out.parent.mkdir(exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else {}
    key = (
        args.model if args.bn_momentum is None else f"{args.model}@bn{args.bn_momentum}"
    )
    key += "@nohealth" if args.no_health else ""
    summary["health_monitor"] = not args.no_health
    rows[key] = summary
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
