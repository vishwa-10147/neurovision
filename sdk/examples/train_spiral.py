"""A standard PyTorch training loop that streams to NeuroVision.

Trains a small MLP on a synthetic 3-arm spiral (no downloads) and logs loss,
accuracy, and post-activation values of two hidden layers after every epoch.
Open NeuroVision → Live runs while it trains.

    cd sdk/examples
    python train_spiral.py --url http://127.0.0.1:8001
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
    args = parser.parse_args()

    torch.manual_seed(0)
    x, y = spiral(samples=900, classes=3, noise=0.4, seed=0)
    split = int(len(x) * 0.8)
    x_train, y_train, x_val, y_val = x[:split], y[:split], x[split:], y[split:]

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    # Post-activation values for one fixed validation example.
    probe = nv.ActivationProbe(
        model, {"hidden_1 (ReLU)": model.act1, "hidden_2 (tanh)": model.act2}, x_val[:1]
    )

    with nv.start_run("Spiral MLP", total_epochs=args.epochs, url=args.url) as run:
        step = 0
        for epoch in range(1, args.epochs + 1):
            correct = total = 0
            running = 0.0
            for i in torch.randperm(len(x_train)).split(32):
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
            )
            if epoch % 25 == 0:
                print(
                    f"epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
                )
            time.sleep(args.delay)
    print(f"done; sent {run.sent} steps, dropped {run.dropped}")


if __name__ == "__main__":
    main()
