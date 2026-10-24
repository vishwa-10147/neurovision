"""Shared training loop for the NeuroVision demos.

Every demo trains a real PyTorch model and streams it to Live Runs with:
- the model's graph (exported to ONNX and linked to the run), colored live
  by activation size, gradient norm or dead units;
- loss, accuracy and validation metrics each epoch;
- post-activation values of chosen layers for one fixed example;
- training-health statistics and alerts;
- CPU/GPU use, step time, throughput and ETA;
- pause, resume, stop and learning-rate control from the dashboard.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neurovision_client as nv  # noqa: E402


def arguments(
    description: str,
    *,
    epochs: int,
    lr: float,
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """Common options; `extra` adds a demo's own (e.g. layer sizes)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--name", help="run name shown in Live runs")
    parser.add_argument("--url", help="NeuroVision backend (default $NEUROVISION_URL)")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr, help="Adam learning rate")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--delay", type=float, default=0.0, help="seconds to pause per epoch"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument(
        "--no-graph", action="store_true", help="do not link the model's graph"
    )
    if extra is not None:
        extra(parser)
    return parser.parse_args()


def device_for(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def train_live(
    name: str,
    model: nn.Module,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    args: argparse.Namespace,
    probe_layers: list[str],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    classify: bool = True,
) -> None:
    """Train `model` on (x_train, y_train, x_val, y_val), streaming to NeuroVision.

    `classify` adds accuracy metrics; otherwise only losses are reported (for
    example an autoencoder, whose targets are its inputs).
    """
    device = device_for(args.device)
    model = model.to(device)
    x_train, y_train, x_val, y_val = (t.to(device) for t in data)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    probe = nv.ActivationProbe.by_name(model, probe_layers, x_val[:1])
    monitor = nv.HealthMonitor(model, optimizer=optimizer, probe_batch=x_val[:64])

    def evaluate() -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            out = model(x_val)
            metrics = {"val_loss": loss_fn(out, y_val).item()}
            if classify:
                metrics["val_accuracy"] = (out.argmax(1) == y_val).float().mean().item()
        model.train()
        return metrics

    started = time.perf_counter()
    with nv.start_run(
        args.name or name,
        total_epochs=args.epochs,
        url=args.url,
        model=None if args.no_graph else model,
        example_input=x_val[:1],
        control=True,
        optimizer=optimizer,
    ) as run:
        step = 0
        for epoch in range(1, args.epochs + 1):
            seen = correct = 0
            running = 0.0
            for index in torch.randperm(len(x_train), device=device).split(args.batch):
                if not run.poll():  # Applies dashboard commands; waits while paused.
                    break
                xb, yb = x_train[index], y_train[index]
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                step += 1
                running += loss.item() * len(xb)
                seen += len(xb)
                if classify:
                    correct += (out.argmax(1) == yb).sum().item()
            if seen == 0:
                print(f"stopped from NeuroVision before epoch {epoch}")
                break
            metrics = {"loss": running / seen, **evaluate()}
            if classify:
                metrics["accuracy"] = correct / seen
            run.log(
                epoch=epoch,
                step=step,
                metrics=metrics,
                activations=probe.capture(),
                health=monitor.collect(),
                samples=seen,
            )
            if run.should_stop:
                print(f"stopped from NeuroVision at epoch {epoch}")
                break
            if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
                shown = ", ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                print(f"epoch {epoch}/{args.epochs}: {shown}", flush=True)
            time.sleep(args.delay)
    print(
        f"done in {time.perf_counter() - started:.1f} s on {device}; "
        f"sent {run.sent} steps, dropped {run.dropped}"
    )
