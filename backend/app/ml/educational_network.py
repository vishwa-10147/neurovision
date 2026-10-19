"""Configurable feedforward network with explicit activation capture.

Each layer's values are recorded as the forward pass computes them, so the
values the UI shows are the ones the model actually produced.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

from app.schemas.architecture import DenseNetworkConfig

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}


@dataclass
class LayerTrace:
    """Values for one layer: `pre` is W·x + b, `post` is after the activation."""

    pre: torch.Tensor | None
    post: torch.Tensor


class EducationalNetwork(nn.Module):
    def __init__(self, config: DenseNetworkConfig) -> None:
        super().__init__()
        sizes = [
            config.input_units,
            *(h.units for h in config.hidden_layers),
            config.output_units,
        ]
        activations = [h.activation for h in config.hidden_layers] + [
            config.output_activation
        ]
        # Graph node ids, matching app/services/educational_graph.py.
        self.node_ids = [
            f"hidden_{i}" for i in range(1, len(config.hidden_layers) + 1)
        ] + ["output"]
        self.activation_names = activations
        # Seeded initialization without disturbing the global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.layers = nn.ModuleList(
                nn.Linear(a, b) for a, b in zip(sizes, sizes[1:], strict=False)
            )
        self.input_units = config.input_units
        self.output_units = config.output_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits: output-layer values before its activation, as used in training."""
        for layer, name in zip(
            self.layers[:-1], self.activation_names[:-1], strict=True
        ):
            x = _apply(name, layer(x))
        return self.layers[-1](x)

    def trace(
        self, x: torch.Tensor, capture: Iterable[str] | None = None
    ) -> dict[str, LayerTrace]:
        """Run a forward pass and record every (or each captured) layer's values."""
        wanted = None if capture is None else set(capture)
        traces: dict[str, LayerTrace] = {}
        if wanted is None or "input" in wanted:
            traces["input"] = LayerTrace(pre=None, post=x)
        for layer, name, node_id in zip(
            self.layers, self.activation_names, self.node_ids, strict=True
        ):
            pre = layer(x)
            x = _apply(name, pre)
            if wanted is None or node_id in wanted:
                traces[node_id] = LayerTrace(pre=pre, post=x)
        return traces

    def parameter_tensors(self) -> dict[str, torch.Tensor]:
        """Weights and biases keyed by tensor id, e.g. "hidden_1.weight"."""
        tensors: dict[str, torch.Tensor] = {}
        for node_id, layer in zip(self.node_ids, self.layers, strict=True):
            tensors[f"{node_id}.weight"] = layer.weight
            tensors[f"{node_id}.bias"] = layer.bias
        return tensors


def _apply(name: str | None, x: torch.Tensor) -> torch.Tensor:
    return x if name is None else _ACTIVATIONS[name](x)
