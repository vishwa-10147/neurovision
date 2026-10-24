"""A standard PyTorch training loop that streams to NeuroVision.

Trains a small MLP on a synthetic 3-arm spiral (no downloads) and logs loss,
accuracy, post-activation values of two hidden layers, and training-health
statistics after every epoch. Open NeuroVision → Live runs while it trains:
the network's graph is shown colored by each layer's activity, health
alerts appear there, and you can pause, resume, stop, save a checkpoint, or
change the learning rate from the dashboard.

    cd sdk/examples
    python train_spiral.py --url http://127.0.0.1:8001
    python train_spiral.py --lr 0.5      # too high: watch the alerts
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neurovision_client as nv  # noqa: E402


def spiral(samples: int, classes: int, noise: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    per_class = samples // classes
    xs, ys = [], []
    for c in range(classes):
        r = torch.linspace(0.05, 1.0, per_class)
        t = r * 4 + c * 2 * math.pi / classes
        t = t + torch.randn(per_class, generator=generator) * noise
        xs.append(torch.stack([r * torch.sin(t), r * torch.cos(t)], dim=1))
        ys.append(torch.full((per_class,), c))
    x, y = torch.cat(xs), torch.cat(ys)
    order = torch.randperm(len(x), generator=generator)
    return x[order], y[order]


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.act2 = nn.Tanh()
        self.out = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act2(self.fc2(self.act1(self.fc1(x)))))


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y).item()
        accuracy = (logits.argmax(1) == y).float().mean().item()
    model.train()
    return loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="NeuroVision backend (default $NEUROVISION_URL)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--delay", type=float, default=0.05, help="seconds to pause per epoch"
    )
    parser.add_argument("--lr", type=float, default=0.003, help="Adam learning rate")
    parser.add_argument(
        "--name", default="Spiral MLP", help="run name shown in Live runs"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    x, y = spiral(samples=900, classes=3, noise=0.4, seed=0)
    split = int(len(x) * 0.8)
    x_train, y_train, x_val, y_val = x[:split], y[:split], x[split:], y[split:]

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    # Post-activation values for one fixed validation example, labeled by
    # module name so Live Runs can place them on the model's graph.
    probe = nv.ActivationProbe.by_name(model, ["act1", "act2"], x_val[:1])
    # Gradient, weight, update and activation statistics for health alerts.
    monitor = nv.HealthMonitor(model, optimizer=optimizer, probe_batch=x_val[:64])
    checkpoints = Path(__file__).resolve().parent / "checkpoints"
    epoch = 0

    def save_checkpoint() -> str:
        checkpoints.mkdir(exist_ok=True)
        path = checkpoints / f"spiral-epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, path)
        return path.name

    with (
        nv.start_run(
            args.name,
            total_epochs=args.epochs,
            url=args.url,
            model=model,  # Linked: Live Runs shows this network's graph.
            example_input=x_val[:1],
            control=True,
            optimizer=optimizer,
            on_checkpoint=save_checkpoint,
        ) as run
    ):
        step = 0
        for epoch in range(1, args.epochs + 1):
            correct = total = 0
            running = 0.0
            for i in torch.randperm(len(x_train)).split(32):
                # Apply dashboard commands; blocks while paused.
                if not run.poll():
                    break
                xb, yb = x_train[i], y_train[i]
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                step += 1
                running += loss.item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(xb)
            if total == 0:
                # Stopped before this epoch's first batch: nothing to log.
                print(f"stopped from NeuroVision before epoch {epoch}")
                break
            val_loss, val_accuracy = evaluate(model, x_val, y_val)
            run.log(
                epoch=epoch,
                step=step,
                metrics={
                    "loss": running / total,
                    "accuracy": correct / total,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                activations=probe.capture(),
                health=monitor.collect(),
            )
            if run.should_stop:
                print(f"stopped from NeuroVision at epoch {epoch}")
                break
            if epoch % 25 == 0:
                print(
                    f"epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
                )
            time.sleep(args.delay)
    print(f"done; sent {run.sent} steps, dropped {run.dropped}")


if __name__ == "__main__":
    main()
