"""Demo: an autoencoder squeezing handwritten digits through 8 numbers.

Real data with no download: scikit-learn's digit images, flattened to 64
pixels. The encoder compresses each image to an 8-value code and the decoder
rebuilds it; the loss is the reconstruction error. Live Runs shows the
network's graph colored by layer activity, the 8 code values of one fixed
digit changing as training goes, health alerts and system use.

    pip install scikit-learn
    python sdk/examples/train_autoencoder.py --url http://127.0.0.1:8001
"""

import argparse

import torch
from demo_common import arguments, train_live
from torch import nn
from train_digits_cnn import digits


class Autoencoder(nn.Module):
    def __init__(self, code: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, code), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def main() -> None:
    def options(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--code", type=int, default=8, help="size of the bottleneck code"
        )

    args = arguments(__doc__.splitlines()[0], epochs=60, lr=0.003, extra=options)
    torch.manual_seed(0)
    x_train, _, x_val, _ = digits()
    x_train, x_val = x_train.flatten(1), x_val.flatten(1)
    train_live(
        f"Digits autoencoder (64 → {args.code} → 64)",
        Autoencoder(args.code),
        (x_train, x_train, x_val, x_val),
        args=args,
        probe_layers=["encoder.1", "encoder.3", "decoder.1"],
        loss_fn=nn.MSELoss(),
        classify=False,
    )


if __name__ == "__main__":
    main()
