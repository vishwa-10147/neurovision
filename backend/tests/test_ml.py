"""Numerical tests for the educational network, datasets, and trainer."""

import json
import math
import threading
from typing import Any

import pytest
import torch

from app.ml.datasets import DatasetError, make_dataset
from app.ml.educational_network import EducationalNetwork
from app.ml.trainer import EpochReport, TrainingDiverged, train
from app.schemas.architecture import DenseNetworkConfig
from app.schemas.export import SHARED_SCHEMAS_DIR
from app.schemas.training import TrainingConfig

EXAMPLE: dict[str, Any] = json.loads(
    (SHARED_SCHEMAS_DIR / "examples" / "dense-network.config.json").read_text(
        encoding="utf-8"
    )
)  # 2 -> 8 (relu) -> 6 (tanh) -> 2


def network(**changes: Any) -> EducationalNetwork:
    return EducationalNetwork(DenseNetworkConfig.model_validate({**EXAMPLE, **changes}))


def dataset(name: str = "moons", **changes: Any):  # type: ignore[no-untyped-def]
    options = {
        "samples": 300,
        "noise": 0.1,
        "input_units": 2,
        "output_units": 2,
        "validation_split": 0.2,
        "seed": 0,
        **changes,
    }
    return make_dataset(name, **options)  # type: ignore[arg-type]


def run(
    net: EducationalNetwork, data, config: TrainingConfig, cancel=None
) -> tuple[bool, list[EpochReport], dict[str, torch.Tensor]]:  # type: ignore[no-untyped-def]
    reports: list[EpochReport] = []
    grads: dict[str, torch.Tensor] = {}
    finished = train(
        net,
        data,
        config,
        model_lock=threading.Lock(),
        cancel=cancel or threading.Event(),
        on_epoch=reports.append,
        on_gradients=grads.update,
    )
    return finished, reports, grads


class TestNetwork:
    def test_trace_matches_a_manual_forward_pass(self) -> None:
        net = network()
        x = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
        traces = net.trace(x)
        assert list(traces) == ["input", "hidden_1", "hidden_2", "output"]

        h1 = torch.relu(x @ net.layers[0].weight.T + net.layers[0].bias)
        h2 = torch.tanh(h1 @ net.layers[1].weight.T + net.layers[1].bias)
        logits = h2 @ net.layers[2].weight.T + net.layers[2].bias
        assert torch.allclose(traces["hidden_1"].post, h1)
        assert torch.allclose(traces["hidden_2"].post, h2)
        assert torch.allclose(traces["output"].pre, logits)
        # No output activation configured: outputs are the raw logits.
        assert torch.allclose(traces["output"].post, logits)
        assert torch.allclose(net(x), logits)

    def test_shapes_follow_the_config(self) -> None:
        traces = network().trace(torch.zeros(5, 2))
        assert {k: tuple(v.post.shape) for k, v in traces.items()} == {
            "input": (5, 2),
            "hidden_1": (5, 8),
            "hidden_2": (5, 6),
            "output": (5, 2),
        }

    def test_output_activation_applies_after_logits(self) -> None:
        net = network(output_activation="sigmoid")
        out = net.trace(torch.randn(3, 2))["output"]
        assert torch.allclose(out.post, torch.sigmoid(out.pre))

    def test_capture_limits_recorded_layers(self) -> None:
        traces = network().trace(torch.zeros(1, 2), capture=["hidden_2"])
        assert list(traces) == ["hidden_2"]

    def test_initialization_is_seeded_and_isolated(self) -> None:
        torch.manual_seed(123)
        expected_next = torch.rand(1)
        torch.manual_seed(123)
        a, b, c = network(seed=7), network(seed=7), network(seed=8)
        # Building networks does not consume the global random stream.
        assert torch.equal(torch.rand(1), expected_next)
        for name, tensor in a.parameter_tensors().items():
            assert torch.equal(tensor, b.parameter_tensors()[name])
        assert not torch.equal(a.layers[0].weight, c.layers[0].weight)

    def test_parameter_ids_and_shapes(self) -> None:
        shapes = {k: tuple(v.shape) for k, v in network().parameter_tensors().items()}
        assert shapes == {
            "hidden_1.weight": (8, 2),
            "hidden_1.bias": (8,),
            "hidden_2.weight": (6, 8),
            "hidden_2.bias": (6,),
            "output.weight": (2, 6),
            "output.bias": (2,),
        }


class TestDatasets:
    @pytest.mark.parametrize("name", ["moons", "circles", "spiral"])
    def test_same_seed_same_points(self, name: str) -> None:
        a, b = dataset(name), dataset(name)
        assert torch.equal(a.x_train, b.x_train)
        assert torch.equal(a.y_val, b.y_val)
        assert not torch.equal(a.x_train, dataset(name, seed=1).x_train)

    def test_split_and_balance(self) -> None:
        data = dataset(samples=301, validation_split=0.2)
        assert data.x_val.shape[0] == 60
        assert data.x_train.shape[0] == 241
        labels = torch.cat([data.y_train, data.y_val])
        assert torch.bincount(labels).tolist() == [151, 150]

    @pytest.mark.parametrize(("inputs", "width"), [(1, 1), (2, 2), (5, 5)])
    def test_features_match_input_units(self, inputs: int, width: int) -> None:
        assert dataset(input_units=inputs).x_train.shape[1] == width

    def test_one_output_unit_means_binary(self) -> None:
        assert dataset(output_units=1).num_classes == 2

    def test_spiral_has_one_arm_per_output(self) -> None:
        data = dataset("spiral", output_units=4)
        assert data.num_classes == 4
        assert set(torch.cat([data.y_train, data.y_val]).tolist()) == {0, 1, 2, 3}

    def test_two_class_datasets_reject_more_outputs(self) -> None:
        with pytest.raises(DatasetError, match="moons dataset has 2 classes"):
            dataset("moons", output_units=3)


class TestTrainer:
    config = TrainingConfig(epochs=40, learning_rate=0.03, seed=0)

    def test_loss_is_finite_and_the_model_learns(self) -> None:
        finished, reports, _ = run(network(), dataset(), self.config)
        assert finished
        assert len(reports) == 40
        losses = [r.metrics.loss for r in reports]
        assert all(math.isfinite(v) for v in losses)
        assert losses[-1] < losses[0] * 0.5
        assert reports[-1].metrics.val_accuracy >= 0.9

    def test_training_is_deterministic(self) -> None:
        config = TrainingConfig(epochs=5, seed=3)
        _, first, _ = run(network(), dataset(), config)
        _, second, _ = run(network(), dataset(), config)
        assert [r.metrics for r in first] == [r.metrics for r in second]

    def test_single_logit_binary_training(self) -> None:
        net = network(output_units=1)
        _, reports, _ = run(net, dataset(output_units=1), self.config)
        assert reports[-1].metrics.val_accuracy >= 0.85

    def test_gradients_are_captured_per_parameter(self) -> None:
        _, _, grads = run(network(), dataset(), TrainingConfig(epochs=1))
        assert set(grads) == {f"{k}.grad" for k in network().parameter_tensors()}
        assert grads["hidden_1.weight.grad"].shape == (8, 2)

    def test_captured_activations_are_measured_values(self) -> None:
        net = network()
        data = dataset()
        config = TrainingConfig(epochs=1, capture_layers=["hidden_1", "output"])
        _, reports, _ = run(net, data, config)
        expected = net.trace(data.x_val[:1])["hidden_1"].post[0].tolist()
        assert reports[-1].activations["hidden_1"] == pytest.approx(expected)
        assert len(reports[-1].activations["output"]) == 2

    def test_cancellation_stops_early(self) -> None:
        cancel = threading.Event()
        reports: list[EpochReport] = []

        def on_epoch(report: EpochReport) -> None:
            reports.append(report)
            if report.metrics.epoch == 2:
                cancel.set()

        finished = train(
            network(),
            dataset(),
            TrainingConfig(epochs=50),
            model_lock=threading.Lock(),
            cancel=cancel,
            on_epoch=on_epoch,
            on_gradients=lambda _: None,
        )
        assert finished is False
        assert len(reports) == 2

    def test_divergence_is_reported_clearly(self) -> None:
        net = network()
        with torch.no_grad():
            net.layers[0].weight.fill_(float("inf"))
        with pytest.raises(TrainingDiverged, match="Lower the learning rate"):
            run(net, dataset(), TrainingConfig(epochs=1))
