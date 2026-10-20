"""Running imported ONNX models: validation, real values, limits, isolation."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

from app.main import create_app
from app.ml.onnx_samples import build_sample

F32 = TensorProto.FLOAT


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    # One app (and one inference worker) for the module: starting a worker
    # process takes about a second.
    with TestClient(create_app()) as test_client:
        yield test_client


def import_sample(client: TestClient, name: str) -> dict[str, Any]:
    response = client.post(f"/api/v1/import/onnx/samples/{name}")
    assert response.status_code == 201
    return dict(response.json())


def upload(client: TestClient, model: onnx.ModelProto) -> dict[str, Any]:
    response = client.post(
        "/api/v1/import/onnx",
        params={"filename": "test.onnx"},
        content=model.SerializeToString(),
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


def run(client: TestClient, model_id: str, **body: Any) -> Any:
    return client.post(f"/api/v1/models/{model_id}/run", json=body)


def mlp_reference(x: np.ndarray) -> dict[str, np.ndarray]:
    """The MLP sample computed by hand from its own weights."""
    model = build_sample("mlp")
    assert model is not None
    w = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    h1 = x @ w["w1"].T + w["b1"]
    a1 = np.maximum(h1, 0)
    logits = a1 @ w["w2"].T + w["b2"]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return {
        "h1": h1,
        "a1": a1,
        "logits": logits,
        "probs": exp / exp.sum(axis=1, keepdims=True),
    }


class TestRunningModels:
    def test_capabilities_and_input_specs(self, client: TestClient) -> None:
        body = import_sample(client, "mlp")
        caps = body["graph"]["model"]["capabilities"]
        assert caps["inference"] is True and caps["activations"] is True
        assert caps["training"] is False
        assert body["report"]["inputs"] == [
            {"name": "x", "node_id": "input_0", "dtype": "float", "shape": ["batch", 4]}
        ]
        assert body["report"]["inference_blockers"] == []

    def test_outputs_and_intermediates_match_a_hand_computation(
        self, client: TestClient
    ) -> None:
        model_id = import_sample(client, "mlp")["model_id"]
        x = np.array([[0.5, -1.0, 2.0, 0.0], [1.0, 1.0, -0.5, 0.25]], dtype=np.float32)
        response = run(
            client,
            model_id,
            inputs={"x": {"shape": [2, 4], "values": x.reshape(-1).tolist()}},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        expected = mlp_reference(x)

        [probs] = body["outputs"]
        assert (
            probs["name"] == "probs"
            and probs["shape"] == [2, 3]
            and probs["node_id"] == "node_3"
        )
        assert probs["values"] == pytest.approx(
            expected["probs"].reshape(-1).tolist(), abs=1e-5
        )
        by_name = {t["name"]: t for t in body["intermediates"]}
        assert set(by_name) == {"h1", "a1", "logits"}
        assert by_name["a1"]["node_id"] == "node_1"
        assert by_name["a1"]["values"] == pytest.approx(
            expected["a1"].reshape(-1).tolist(), abs=1e-5
        )
        assert by_name["a1"]["summary"]["min"] >= 0  # ReLU output
        assert body["runtime"].startswith("onnxruntime ")
        assert body["elapsed_ms"] >= 0

    def test_generated_inputs_are_deterministic(self, client: TestClient) -> None:
        model_id = import_sample(client, "mlp")["model_id"]
        first = run(client, model_id, inputs={"x": {"shape": [3, 4], "seed": 7}}).json()
        again = run(client, model_id, inputs={"x": {"shape": [3, 4], "seed": 7}}).json()
        other = run(client, model_id, inputs={"x": {"shape": [3, 4], "seed": 8}}).json()
        assert first["outputs"] == again["outputs"]
        assert first["outputs"] != other["outputs"]
        zeros = run(
            client, model_id, inputs={"x": {"shape": [1, 4], "fill": "zeros"}}
        ).json()
        expected = mlp_reference(np.zeros((1, 4), dtype=np.float32))["probs"]
        assert zeros["outputs"][0]["values"] == pytest.approx(
            expected.reshape(-1).tolist(), abs=1e-5
        )

    def test_large_tensors_are_bounded(self, client: TestClient) -> None:
        model_id = import_sample(client, "cnn")["model_id"]
        body = run(client, model_id, inputs={"image": {"shape": [1, 1, 28, 28]}}).json()
        conv1 = next(t for t in body["intermediates"] if t["node_id"] == "node_0")
        assert conv1["shape"] == [1, 8, 28, 28] and conv1["count"] == 6272
        assert len(conv1["values"]) == 4096 and conv1["truncated"] is True
        assert body["outputs"][0]["shape"] == [1, 10]

    def test_capture_selects_operators(self, client: TestClient) -> None:
        model_id = import_sample(client, "cnn")["model_id"]
        body = run(
            client,
            model_id,
            inputs={"image": {"shape": [2, 1, 28, 28]}},
            capture=["node_2"],
        ).json()
        assert [t["node_id"] for t in body["intermediates"]] == ["node_2"]
        assert body["intermediates"][0]["shape"] == [2, 8, 14, 14]

    def test_a_real_pytorch_export_runs(self, client: TestClient) -> None:
        torch = pytest.importorskip("torch")
        import io

        net = torch.nn.Sequential(
            torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)
        ).eval()
        buffer = io.BytesIO()
        torch.onnx.export(
            net, torch.zeros(1, 3), buffer, input_names=["features"], dynamo=False
        )
        model_id = client.post(
            "/api/v1/import/onnx",
            params={"filename": "net.onnx"},
            content=buffer.getvalue(),
        ).json()["model_id"]
        x = [[0.1, -0.2, 0.3]]
        body = run(
            client, model_id, inputs={"features": {"shape": [1, 3], "values": x[0]}}
        ).json()
        with torch.no_grad():
            expected = net(torch.tensor(x)).reshape(-1).tolist()
        assert body["outputs"][0]["values"] == pytest.approx(expected, abs=1e-5)


class TestValidation:
    @pytest.mark.parametrize(
        ("inputs", "field", "message"),
        [
            ({}, "body.inputs.x", "Provide a value"),
            (
                {"x": {"shape": [1, 4]}, "y": {"shape": [1]}},
                "body.inputs.y",
                "no input named 'y'",
            ),
            ({"x": {"shape": [4]}}, "body.inputs.x.shape", "Expected 2 dimensions"),
            ({"x": {"shape": [1, 5]}}, "body.inputs.x.shape.1", "must be 4"),
            (
                {"x": {"shape": [1, 4], "values": [1, 2, 3]}},
                "body.inputs.x.values",
                "Expected 4 values",
            ),
        ],
    )
    def test_inputs_must_match_the_model(
        self, client: TestClient, inputs: dict[str, Any], field: str, message: str
    ) -> None:
        model_id = import_sample(client, "mlp")["model_id"]
        response = run(client, model_id, inputs=inputs)
        assert response.status_code == 422
        detail = response.json()["error"]["details"][0]
        assert detail["field"] == field and message in detail["message"]

    def test_total_input_size_is_limited(self, client: TestClient) -> None:
        model_id = import_sample(client, "cnn")["model_id"]
        # Every dimension is allowed, but 1276 × 1 × 28 × 28 = 1,000,384 values is not.
        response = run(client, model_id, inputs={"image": {"shape": [1276, 1, 28, 28]}})
        assert response.status_code == 422
        detail = response.json()["error"]["details"][0]
        assert detail["field"] == "body.inputs.image.shape"
        assert "at most 1,000,000" in detail["message"]

    def test_schema_level_limits(self, client: TestClient) -> None:
        model_id = import_sample(client, "mlp")["model_id"]
        assert run(client, model_id, inputs={"x": {"shape": [1, 0]}}).status_code == 422
        assert (
            run(client, model_id, inputs={"x": {"shape": [1] * 9}}).status_code == 422
        )
        assert (
            run(
                client, model_id, inputs={"x": {"shape": [1, 4], "fill": "chaos"}}
            ).status_code
            == 422
        )

    def test_integer_inputs_must_be_whole_and_in_range(
        self, client: TestClient
    ) -> None:
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Cast", ["ids"], ["y"], to=F32)],
                "ints",
                [helper.make_tensor_value_info("ids", TensorProto.INT8, [2])],
                [helper.make_tensor_value_info("y", F32, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.ir_version = 8
        model_id = upload(client, model)["model_id"]
        assert (
            run(
                client, model_id, inputs={"ids": {"shape": [2], "values": [1.5, 2]}}
            ).status_code
            == 422
        )
        too_big = run(
            client, model_id, inputs={"ids": {"shape": [2], "values": [1, 300]}}
        )
        assert (
            too_big.status_code == 422
            and "-128 to 127" in too_big.json()["error"]["details"][0]["message"]
        )
        ok = run(client, model_id, inputs={"ids": {"shape": [2], "values": [-3, 7]}})
        assert ok.status_code == 200 and ok.json()["outputs"][0]["values"] == [
            -3.0,
            7.0,
        ]

    def test_unknown_capture_target(self, client: TestClient) -> None:
        model_id = import_sample(client, "mlp")["model_id"]
        response = run(
            client, model_id, inputs={"x": {"shape": [1, 4]}}, capture=["node_99"]
        )
        assert response.status_code == 422
        assert response.json()["error"]["details"][0]["field"] == "body.capture.0"


class TestUnsupportedAndFailures:
    def test_custom_operators_block_running_but_not_inspection(
        self, client: TestClient
    ) -> None:
        body = import_sample(client, "custom-op")
        assert body["graph"]["model"]["capabilities"]["inference"] is False
        assert any("Scale" in b for b in body["report"]["inference_blockers"])
        response = run(client, body["model_id"], inputs={"x": {"shape": [1, 4]}})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "inference_unsupported"
        # Inspection still works.
        assert client.get(f"/api/v1/models/{body['model_id']}/graph").status_code == 200

    def test_runtime_errors_are_returned_safely(self, client: TestClient) -> None:
        # Valid ONNX that ONNX Runtime rejects when loading: mismatched MatMul shapes.
        w = numpy_helper.from_array(np.ones((3, 2), dtype=np.float32), "w")
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("MatMul", ["x", "w"], ["y"])],
                "bad_matmul",
                [helper.make_tensor_value_info("x", F32, [1, 4])],
                [helper.make_tensor_value_info("y", F32, None)],
                [w],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.ir_version = 8
        model_id = upload(client, model)["model_id"]
        response = run(client, model_id, inputs={"x": {"shape": [1, 4]}})
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "inference_failed"
        assert error["message"].startswith("ONNX Runtime could not run this model:")
        assert "Traceback" not in response.text
        # The worker survives a failed run.
        ok = run(
            client,
            import_sample(client, "mlp")["model_id"],
            inputs={"x": {"shape": [1, 4]}},
        )
        assert ok.status_code == 200

    def test_educational_models_use_forward_instead(self, client: TestClient) -> None:
        config = {"name": "n", "input_units": 2, "hidden_layers": [], "output_units": 2}
        model_id = client.post("/api/v1/models/educational", json=config).json()[
            "model_id"
        ]
        response = run(client, model_id, inputs={})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "capability_unavailable"


class TestIsolation:
    def test_timeouts_stop_the_worker_and_it_recovers(self, client: TestClient) -> None:
        runtime = client.app.state.onnx_runtime  # type: ignore[attr-defined]
        model_id = import_sample(client, "cnn")["model_id"]
        original = runtime.timeout_s
        runtime.timeout_s = 0.0001
        try:
            response = run(
                client, model_id, inputs={"image": {"shape": [64, 1, 28, 28]}}
            )
        finally:
            runtime.timeout_s = original
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "inference_timeout"
        recovered = run(client, model_id, inputs={"image": {"shape": [1, 1, 28, 28]}})
        assert recovered.status_code == 200
        assert any(
            "started the inference worker" in n for n in recovered.json()["notes"]
        )

    def test_a_killed_worker_is_replaced(self, client: TestClient) -> None:
        runtime = client.app.state.onnx_runtime  # type: ignore[attr-defined]
        model_id = import_sample(client, "mlp")["model_id"]
        assert run(client, model_id, inputs={"x": {"shape": [1, 4]}}).status_code == 200
        runtime._process.kill()  # Simulate a crash inside ONNX Runtime.
        runtime._process.join(5)
        response = run(client, model_id, inputs={"x": {"shape": [1, 4]}})
        assert response.status_code == 200

    def test_the_api_process_never_imports_onnxruntime(
        self, client: TestClient
    ) -> None:
        import sys

        model_id = import_sample(client, "mlp")["model_id"]
        assert run(client, model_id, inputs={"x": {"shape": [1, 4]}}).status_code == 200
        assert "onnxruntime" not in sys.modules
