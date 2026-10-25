"""Train a tiny PyTorch Lightning model with one NeuroVision callback."""

import argparse

import lightning.pytorch as pl
import torch
from neurovision_callbacks import LightningCallback
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Classifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
        self.example_input_array = torch.zeros(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        del batch_idx
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        del batch_idx
        x, y = batch
        self.log("val_loss", nn.functional.cross_entropy(self(x), y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(0)
    x = torch.randn(128, 2)
    y = (x[:, 0] + x[:, 1] > 0).long()
    train = TensorDataset(x[:96], y[:96])
    validation = TensorDataset(x[96:], y[96:])
    callback = LightningCallback(
        url=args.url,
        control=True,
        activation_layers=["net.1"],
        probe_batch=x[96:97],
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=False,
        enable_checkpointing=False,
        callbacks=[callback],
    )
    trainer.fit(
        Classifier(),
        DataLoader(train, batch_size=16),
        DataLoader(validation, batch_size=32),
    )


if __name__ == "__main__":
    main()
