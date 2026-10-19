import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.architecture import DenseNetworkConfig
from app.schemas.export import SHARED_SCHEMAS_DIR
from app.schemas.graph import ModelGraph
from app.services.educational_graph import build_dense_graph
from app.services.model_repository import InMemoryModelRepository

EXAMPLE_CONFIG: dict[str, Any] = json.loads(
    (SHARED_SCHEMAS_DIR / "examples" / "dense-network.config.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _config(**changes: Any) -> dict[str, Any]:
    return {**EXAMPLE_CONFIG, **changes}


def test_graph_structure_and_parameter_counts() -> None:
    # 2 -> 8 (relu) -> 6 (tanh) -> 2
    graph = build_dense_graph(DenseNetworkConfig.model_validate(EXAMPLE_CONFIG), "m1")
    assert [n.id for n in graph.nodes] == ["input", "hidden_1", "hidden_2", "output"]
    assert [(e.source, e.target) for e in graph.edges] == [
        ("input", "hidden_1"),
        ("hidden_1", "hidden_2"),
        ("hidden_2", "output"),
    ]
    counts = [
        n.parameter_summary.parameter_count if n.parameter_summary else None
        for n in graph.nodes
    ]
    assert counts == [None, 2 * 8 + 8, 8 * 6 + 6, 6 * 2 + 2]
    assert graph.model.metadata["educational"]["parameter_count"] == 24 + 54 + 14
    assert graph.nodes[2].attributes["activation"] == "tanh"
    assert graph.nodes[3].attributes["activation"] is None
    assert graph.nodes[2].input_shapes == [["batch", 8]]
    assert graph.nodes[2].output_shapes == [["batch", 6]]


def test_graph_without_hidden_layers() -> None:
    config = DenseNetworkConfig.model_validate(_config(hidden_layers=[]))
    graph = build_dense_graph(config, "m2")
    assert [n.id for n in graph.nodes] == ["input", "output"]
    assert graph.model.metadata["educational"]["parameter_count"] == 2 * 2 + 2


def test_capabilities_are_graph_only_until_pytorch_lands() -> None:
    graph = build_dense_graph(DenseNetworkConfig.model_validate(EXAMPLE_CONFIG), "m")
    caps = graph.model.capabilities
    assert caps.graph is True
    assert not any([caps.inference, caps.training, caps.activations, caps.gradients])


def test_create_then_fetch_graph(client: TestClient) -> None:
    created = client.post("/api/v1/models/educational", json=EXAMPLE_CONFIG)
    assert created.status_code == 201
    body = created.json()
    graph = ModelGraph.model_validate(body["graph"])
    assert graph.model.id == body["model_id"]
    assert graph.model.name == EXAMPLE_CONFIG["name"]

    fetched = client.get(f"/api/v1/models/{body['model_id']}/graph")
    assert fetched.status_code == 200
    assert fetched.json() == body["graph"]


def test_invalid_config_is_rejected_with_field_details(client: TestClient) -> None:
    response = client.post(
        "/api/v1/models/educational",
        json=_config(hidden_layers=[{"units": 99, "activation": "relu"}]),
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["field"] == "body.hidden_layers.0.units"
    assert "32" in error["details"][0]["message"]


def test_unknown_model_returns_404_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/models/edu_missing/graph")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert "edu_missing" in response.json()["error"]["message"]


def test_model_id_path_is_restricted(client: TestClient) -> None:
    response = client.get("/api/v1/models/..%2F..%2Fetc/graph")
    assert response.status_code in (404, 422)


def test_repository_evicts_least_recently_used() -> None:
    repository = InMemoryModelRepository(capacity=2)
    config = DenseNetworkConfig.model_validate(EXAMPLE_CONFIG)
    for model_id in ("a", "b"):
        repository.add(build_dense_graph(config, model_id))
    assert repository.get_graph("a") is not None  # "a" becomes most recent
    repository.add(build_dense_graph(config, "c"))
    assert repository.get_graph("b") is None
    assert repository.get_graph("a") is not None
    assert len(repository) == 2
