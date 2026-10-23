import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.architecture import DenseNetworkConfig
from app.schemas.events import TrainingStep, training_event_adapter
from app.schemas.export import SHARED_SCHEMAS_DIR, build_schemas, render
from app.schemas.graph import ModelGraph
from app.schemas.inference import TensorGridResponse, TensorMapResponse

EXAMPLES = SHARED_SCHEMAS_DIR / "examples"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["dense-network.graph.json", "onnx-import.graph.json"])
def test_example_graphs_are_valid(name: str) -> None:
    ModelGraph.model_validate(_load(EXAMPLES / name))


def test_example_tensor_views_are_valid() -> None:
    grid = TensorGridResponse.model_validate(_load(EXAMPLES / "attention.grid.json"))
    detail = TensorMapResponse.model_validate(_load(EXAMPLES / "attention.map.json"))
    # The detail map is head 1 of the grid, so the two must agree.
    assert detail.values == grid.items[1].values
    assert all(sum(row) == pytest.approx(1, abs=1e-5) for row in detail.values)  # type: ignore[arg-type]


def test_example_event_is_valid() -> None:
    event = training_event_adapter.validate_python(
        _load(EXAMPLES / "training-step.event.json")
    )
    assert isinstance(event, TrainingStep)
    assert event.metrics["loss"] == pytest.approx(0.42)


def test_example_health_and_control_events_are_valid() -> None:
    events = [
        training_event_adapter.validate_python(e)
        for e in _load(EXAMPLES / "run-health.events.json")
    ]
    assert [e.event_type for e in events] == [
        "training.step",
        "run.command",
        "run.alert",
    ]


def test_unknown_dimensions_stay_unknown() -> None:
    graph = ModelGraph.model_validate(_load(EXAMPLES / "onnx-import.graph.json"))
    custom = graph.nodes[1]
    assert custom.unsupported is True
    assert custom.input_shapes == [None]
    assert custom.output_shapes == [["N", None, None, None]]


_INVALID = _load(EXAMPLES / "invalid-cases.json")


@pytest.mark.parametrize("case", _INVALID, ids=[c["description"] for c in _INVALID])
def test_invalid_cases_are_rejected(case: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        if case["kind"] == "graph":
            ModelGraph.model_validate(case["document"])
        elif case["kind"] == "architecture":
            DenseNetworkConfig.model_validate(case["document"])
        else:
            training_event_adapter.validate_python(case["document"])


def test_example_config_is_valid() -> None:
    DenseNetworkConfig.model_validate(_load(EXAMPLES / "dense-network.config.json"))


def test_non_finite_metrics_are_rejected() -> None:
    event = _load(EXAMPLES / "training-step.event.json")
    event["metrics"]["loss"] = float("nan")
    with pytest.raises(ValidationError):
        training_event_adapter.validate_python(event)


def test_integrity_error_names_the_problem() -> None:
    dangling = next(c for c in _INVALID if c["description"].startswith("edge"))
    with pytest.raises(ValidationError, match="unknown node 'missing'"):
        ModelGraph.model_validate(dangling["document"])


@pytest.mark.parametrize("name", list(build_schemas()))
def test_shared_schema_files_match_models(name: str) -> None:
    committed = (SHARED_SCHEMAS_DIR / name).read_text(encoding="utf-8")
    assert committed == render(build_schemas()[name]), (
        f"{name} is out of date; run `python -m app.schemas.export` in backend/"
    )
