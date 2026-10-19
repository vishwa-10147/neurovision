"""Training loop with validation metrics, gradient snapshots, and cancellation."""

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from app.ml.datasets import Dataset
from app.ml.educational_network import EducationalNetwork
from app.schemas.training import EpochMetrics, TrainingConfig


class TrainingDiverged(RuntimeError):
    """The loss stopped being a finite number."""


@dataclass
class EpochReport:
    metrics: EpochMetrics
    step: int
    # Post-activation values of captured layers for one validation example.
    activations: dict[str, list[float]]


class Objective:
    """Loss and accuracy for single-logit binary or multi-class outputs."""

    def __init__(self, output_units: int) -> None:
        self.binary = output_units == 1
        self.loss_fn: nn.Module = (
            nn.BCEWithLogitsLoss() if self.binary else nn.CrossEntropyLoss()
        )

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.binary:
            return self.loss_fn(logits.squeeze(1), y.float())
        return self.loss_fn(logits, y)

    def predictions(self, logits: torch.Tensor) -> torch.Tensor:
        if self.binary:
            return (logits.squeeze(1) > 0).long()
        return logits.argmax(dim=1)

    def accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        return (self.predictions(logits) == y).float().mean().item()


def train(
    network: EducationalNetwork,
    data: Dataset,
    config: TrainingConfig,
    *,
    model_lock: threading.Lock,
    cancel: threading.Event,
    on_epoch: Callable[[EpochReport], None],
    on_gradients: Callable[[dict[str, torch.Tensor]], None],
) -> bool:
    """Train in place. Returns False if cancelled before finishing.

    The model lock is held per batch, so forward passes from the API can
    interleave with training without seeing half-updated weights.
    """
    objective = Objective(network.output_units)
    params = list(network.parameters())
    optimizer: torch.optim.Optimizer = (
        torch.optim.Adam(params, lr=config.learning_rate)
        if config.optimizer == "adam"
        else torch.optim.SGD(params, lr=config.learning_rate, momentum=0.9)
    )
    gen = torch.Generator().manual_seed(config.seed)
    n = data.x_train.shape[0]
    probe = data.x_val[:1]
    step = 0

    for epoch in range(1, config.epochs + 1):
        network.train()
        total_loss, total_correct = 0.0, 0.0
        for batch in torch.randperm(n, generator=gen).split(config.batch_size):
            if cancel.is_set():
                return False
            x, y = data.x_train[batch], data.y_train[batch]
            with model_lock:
                optimizer.zero_grad()
                logits = network(x)
                loss = objective.loss(logits, y)
                if not math.isfinite(loss.item()):
                    raise TrainingDiverged(
                        f"The loss became {loss.item()} in epoch {epoch}. "
                        "Lower the learning rate and try again."
                    )
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(batch)
            total_correct += objective.accuracy(logits.detach(), y) * len(batch)
            step += 1

        # Gradients from the epoch's last batch, for inspection.
        on_gradients(
            {
                f"{tensor_id}.grad": tensor.grad.detach().clone()
                for tensor_id, tensor in network.parameter_tensors().items()
                if tensor.grad is not None
            }
        )

        network.eval()
        with model_lock, torch.no_grad():
            val_logits = network(data.x_val)
            val_loss = objective.loss(val_logits, data.y_val).item()
            val_accuracy = objective.accuracy(val_logits, data.y_val)
            traces = (
                network.trace(probe, config.capture_layers)
                if config.capture_layers
                else {}
            )

        on_epoch(
            EpochReport(
                metrics=EpochMetrics(
                    epoch=epoch,
                    loss=total_loss / n,
                    accuracy=total_correct / n,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                ),
                step=step,
                activations={
                    node_id: t.post[0].tolist() for node_id, t in traces.items()
                },
            )
        )
        if config.epoch_delay_ms and cancel.wait(config.epoch_delay_ms / 1000):
            return False
    return True
