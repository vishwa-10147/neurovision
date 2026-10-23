"""Training-health rules (ADR 0008): each rule fires once when its condition
starts, quotes the measured numbers, and re-arms when the condition clears."""

from typing import Any

from app.schemas.events import StepHealth
from app.services.run_health import PERSIST_STEPS, PLATEAU_STEPS, HealthRules


def health(**fields: Any) -> StepHealth:
    return StepHealth.model_validate(fields)


def kinds(found: list[Any]) -> list[str]:
    return [f.kind for f in found]


def feed(
    rules: HealthRules, steps: list[tuple[dict[str, float], StepHealth | None]]
) -> list[list[str]]:
    return [kinds(rules.observe(m, h)) for m, h in steps]


class TestGradients:
    def test_exploding_gradients_fire_once_and_rearm(self) -> None:
        rules = HealthRules()
        norms = [1.0, 1.1, 0.9, 1.0, 1.2, 50.0, 60.0, 1.0, 55.0]
        found = feed(rules, [({}, health(grad_norm=n)) for n in norms])
        assert found[5] == ["exploding_gradients"]
        assert found[6] == []  # Still exploding: not repeated.
        assert found[7] == []  # Cleared: re-armed.
        assert found[8] == ["exploding_gradients"]

    def test_needs_history_before_judging(self) -> None:
        rules = HealthRules()
        assert feed(rules, [({}, health(grad_norm=n)) for n in (1.0, 100.0, 1.0)]) == [
            [],
            [],
            [],
        ]

    def test_message_quotes_the_numbers(self) -> None:
        rules = HealthRules()
        for n in (2.0, 2.0, 2.0, 2.0, 2.0):
            rules.observe({}, health(grad_norm=n))
        (finding,) = rules.observe({}, health(grad_norm=40.0))
        assert (
            "40" in finding.message
            and "20×" in finding.message
            and "2" in finding.message
        )
        assert finding.severity == "warning" and "clip" in finding.advice

    def test_non_finite_gradients_are_critical(self) -> None:
        rules = HealthRules()
        (finding,) = rules.observe({}, health(nonfinite_grads=12))
        assert finding.kind == "non_finite_gradients" and finding.severity == "critical"
        assert "12" in finding.message

    def test_vanishing_gradients_need_persistence(self) -> None:
        rules = HealthRules()
        step = health(parameters={"fc1": {"grad_norm": 1e-9, "weight_norm": 1.0}})
        found = feed(rules, [({}, step)] * PERSIST_STEPS)
        assert found[:-1] == [[]] * (PERSIST_STEPS - 1)
        assert found[-1] == ["vanishing_gradients"]
        assert rules.observe({}, step) == []


class TestUpdatesAndActivations:
    def test_update_ratio_high_names_the_layer(self) -> None:
        rules = HealthRules()
        (finding,) = rules.observe(
            {}, health(parameters={"conv1": {"update_ratio": 0.5}})
        )
        assert finding.kind == "update_ratio_high" and finding.layer == "conv1"
        assert "5.0e-01" in finding.message

    def test_healthy_update_ratio_is_quiet(self) -> None:
        rules = HealthRules()
        step = health(parameters={"fc": {"update_ratio": 1e-3, "grad_norm": 0.1}})
        assert feed(rules, [({}, step)] * 5) == [[]] * 5

    def test_tiny_updates_are_info(self) -> None:
        rules = HealthRules()
        step = health(parameters={"fc": {"update_ratio": 1e-8}})
        found = feed(rules, [({}, step)] * PERSIST_STEPS)
        assert found[-1] == ["update_ratio_low"]

    def test_dead_and_saturated_units(self) -> None:
        rules = HealthRules()
        step = health(
            activations={
                "relu1": {
                    "kind": "relu",
                    "mean": 0.0,
                    "std": 0.0,
                    "dead_fraction": 0.75,
                },
                "tanh1": {
                    "kind": "tanh",
                    "mean": 0.9,
                    "std": 0.1,
                    "saturated_fraction": 0.6,
                },
            }
        )
        found = [rules.observe({}, step) for _ in range(PERSIST_STEPS)]
        last = found[-1]
        assert {(f.kind, f.layer) for f in last} == {
            ("dead_units", "relu1"),
            ("saturated_units", "tanh1"),
        }
        assert any("75%" in f.message for f in last)


class TestLoss:
    def test_non_finite_loss_is_critical(self) -> None:
        rules = HealthRules()
        (finding,) = rules.observe({}, health(nonfinite_metrics=["loss"]))
        assert finding.kind == "non_finite_loss" and finding.severity == "critical"
        assert "loss" in finding.message

    def test_loss_spike(self) -> None:
        rules = HealthRules()
        found = feed(
            rules, [({"loss": v}, None) for v in (1.0, 0.9, 0.8, 0.8, 0.7, 5.0)]
        )
        assert found[-1] == ["loss_spike"]

    def test_plateau_fires_after_no_improvement(self) -> None:
        rules = HealthRules()
        found = feed(rules, [({"loss": 0.5}, None)] * (PLATEAU_STEPS + 1))
        assert found[-1] == ["loss_plateau"] and all(f == [] for f in found[:-1])
        # Real improvement re-arms it.
        assert rules.observe({"loss": 0.3}, None) == []

    def test_improving_loss_never_plateaus(self) -> None:
        rules = HealthRules()
        found = feed(rules, [({"loss": 1.0 * 0.95**i}, None) for i in range(40)])
        assert all(f == [] for f in found)

    def test_overfitting_when_validation_rises_and_training_falls(self) -> None:
        rules = HealthRules()
        steps = [
            ({"loss": 1.0 - i * 0.1, "val_loss": 0.5 + i * 0.05}, None)
            for i in range(6)
        ]
        found = feed(rules, steps)
        assert found[-1] == ["overfitting"] and all(f == [] for f in found[:-1])

    def test_no_overfitting_when_both_fall(self) -> None:
        rules = HealthRules()
        steps = [
            ({"loss": 1.0 - i * 0.1, "val_loss": 1.0 - i * 0.05}, None)
            for i in range(8)
        ]
        assert all(f == [] for f in feed(rules, steps))
