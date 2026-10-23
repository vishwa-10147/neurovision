"""Demo: a convolutional network learning handwritten digits, live.

Real data with no download: scikit-learn's 1,797 digit images (8×8 grey).
Two convolution blocks and a classifier; Live Runs shows the network's graph
colored by how strongly each layer fires, alongside loss, accuracy, feature
values, health alerts and system use.

    pip install scikit-learn
    python sdk/examples/train_digits_cnn.py --url http://127.0.0.1:8001
"""

import torch
from demo_common import arguments, train_live
from torch import nn


class DigitsCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8×8 → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
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
    args = arguments(__doc__.splitlines()[0], epochs=30, lr=0.002)
    torch.manual_seed(0)
    train_live(
        "Digits CNN",
        DigitsCNN(),
        digits(),
        args=args,
        probe_layers=["features.1", "features.3", "classifier.2"],
    )


if __name__ == "__main__":
    main()
