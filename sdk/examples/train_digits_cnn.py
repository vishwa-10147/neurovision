"""Demo: a convolutional network learning handwritten digits, live.

Real data with no download: scikit-learn's 1,797 digit images (8×8 grey).
Two convolution blocks and a classifier; Live Runs shows the network's graph
colored by how strongly each layer fires, alongside loss, accuracy, feature
values, health alerts and system use.

    pip install scikit-learn
    python sdk/examples/train_digits_cnn.py --url http://127.0.0.1:8001
"""

import argparse

import torch
from demo_common import arguments, train_live
from torch import nn


class DigitsCNN(nn.Module):
    """`blocks` conv + ReLU layers (channels doubling), a 2×2 max-pool, a classifier."""

    def __init__(self, blocks: int = 2, channels: int = 16, hidden: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = 1
        for b in range(blocks):
            layers += [nn.Conv2d(width, channels * 2**b, 3, padding=1), nn.ReLU()]
            width = channels * 2**b
        layers.append(nn.MaxPool2d(2))  # 8×8 → 4×4
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def digits() -> tuple[torch.Tensor, ...]:
    try:
        from sklearn.datasets import load_digits
    except ImportError:
        raise SystemExit(
            "This demo needs scikit-learn: pip install scikit-learn"
        ) from None
    data = load_digits()
    x = torch.tensor(data.images, dtype=torch.float32).unsqueeze(1) / 16.0
    y = torch.tensor(data.target)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    x, y = x[order], y[order]
    split = int(len(x) * 0.8)
    return x[:split], y[:split], x[split:], y[split:]


def main() -> None:
    def options(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--blocks", type=int, default=2, help="convolution layers (1-4)"
        )
        parser.add_argument(
            "--channels", type=int, default=16, help="channels of the first layer"
        )
        parser.add_argument(
            "--hidden", type=int, default=64, help="units in the classifier"
        )

    args = arguments(__doc__.splitlines()[0], epochs=30, lr=0.002, extra=options)
    torch.manual_seed(0)
    train_live(
        "Digits CNN",
        DigitsCNN(args.blocks, args.channels, args.hidden),
        digits(),
        args=args,
        # The ReLU after each convolution, and the classifier's ReLU.
        probe_layers=[f"features.{2 * b + 1}" for b in range(args.blocks)][:7]
        + ["classifier.2"],
    )


if __name__ == "__main__":
    main()
