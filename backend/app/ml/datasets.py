"""Deterministic synthetic classification datasets.

Every dataset is generated from a seed, so the same request always produces
the same points. Points are 2D; networks with more inputs receive extra
uninformative noise features, and networks with one input see only x.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch

DatasetName = Literal["moons", "circles", "spiral"]

# Datasets with a fixed number of classes; "spiral" has one arm per class.
FIXED_CLASSES: dict[str, int] = {"moons": 2, "circles": 2}
DISTRACTOR_STD = 0.5


@dataclass(frozen=True)
class Dataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    num_classes: int


class DatasetError(ValueError):
    """The dataset cannot be built for the requested network shape."""


def classes_for(name: DatasetName, output_units: int) -> int:
    """Classes the dataset will have for a network with `output_units` outputs.

    One output unit means binary classification with a single logit.
    """
    wanted = 2 if output_units == 1 else output_units
    fixed = FIXED_CLASSES.get(name)
    if fixed is not None and wanted != fixed:
        raise DatasetError(
            f"The {name} dataset has {fixed} classes, so the network needs 1 or 2 "
            f"output units (it has {output_units}). Use the spiral dataset for "
            "more classes."
        )
    return wanted


def make_dataset(
    name: DatasetName,
    *,
    samples: int,
    noise: float,
    input_units: int,
    output_units: int,
    validation_split: float,
    seed: int,
) -> Dataset:
    num_classes = classes_for(name, output_units)
    gen = torch.Generator().manual_seed(seed)
    per_class = [
        samples // num_classes + (1 if c < samples % num_classes else 0)
        for c in range(num_classes)
    ]

    points: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for cls, count in enumerate(per_class):
        xy = _points(name, cls, num_classes, count, gen)
        xy = xy + torch.randn(xy.shape, generator=gen) * noise
        points.append(xy)
        labels.append(torch.full((count,), cls, dtype=torch.long))
    x = torch.cat(points)
    y = torch.cat(labels)

    if input_units == 1:
        x = x[:, :1]
    elif input_units > 2:
        extra = (
            torch.randn((x.shape[0], input_units - 2), generator=gen) * DISTRACTOR_STD
        )
        x = torch.cat([x, extra], dim=1)

    order = torch.randperm(x.shape[0], generator=gen)
    x, y = x[order], y[order]
    n_val = max(1, round(x.shape[0] * validation_split))
    return Dataset(
        x_train=x[n_val:].contiguous(),
        y_train=y[n_val:].contiguous(),
        x_val=x[:n_val].contiguous(),
        y_val=y[:n_val].contiguous(),
        num_classes=num_classes,
    )


def _points(
    name: DatasetName, cls: int, num_classes: int, count: int, gen: torch.Generator
) -> torch.Tensor:
    if name == "moons":
        t = torch.rand(count, generator=gen) * math.pi
        if cls == 0:
            return torch.stack([torch.cos(t), torch.sin(t)], dim=1) - torch.tensor(
                [0.5, 0.25]
            )
        return torch.stack(
            [1 - torch.cos(t), 0.5 - torch.sin(t)], dim=1
        ) - torch.tensor([0.5, 0.25])
    if name == "circles":
        t = torch.rand(count, generator=gen) * 2 * math.pi
        radius = 1.0 if cls == 0 else 0.5
        return torch.stack([torch.cos(t), torch.sin(t)], dim=1) * radius
    # Spiral: one arm per class, radius growing along the arm.
    r = torch.linspace(0.05, 1.0, count)
    t = torch.linspace(0.0, 3.0, count) + cls * (2 * math.pi / num_classes)
    return torch.stack([r * torch.sin(t), r * torch.cos(t)], dim=1)
