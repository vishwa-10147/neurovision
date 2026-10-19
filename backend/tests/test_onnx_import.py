"""ONNX import: safe parsing, normalization, limits, and bounded tensors."""

import builtins
import os
import tempfile
from collections.abc import Iterator
from typing import Any

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

from app.core.config import get_settings
from app.main import create_app
from app.ml.onnx_samples import SAMPLES, build_sample
from app.schemas.graph import ModelGraph
from app.services.onnx_importer import MAX_OPERATORS, OnnxImportError, import_onnx

F32 = TensorProto.FLOAT


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def model_bytes(nodes, inputs, outputs, inits=(), opsets=None) -> bytes:  # type: ignore[no-untyped-def]
    graph = helper.make_graph(list(nodes), "test_graph", inputs, outputs, list(inits))
    model = helper.make_model(
        graph, opset_imports=opsets or [helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model.SerializeToString()


def upload(client: TestClient, data: bytes, filename: str = "model.onnx") -> Any:
    return client.post(
        "/api/v1/import/onnx",
        params={"filename": filename},
        content=data,
        headers={"content-type": "application/octet-stream"},
    )


def mlp_bytes() -> bytes:
    sample = build_sample("mlp")
    assert sample is not None
    return sample.SerializeToString()


class TestNormalization:
    def test_mlp_becomes_a_valid_operator_graph(self) -> None:
        result = import_onnx(mlp_bytes(), "m1", "mlp.onnx")
        graph = ModelGraph.model_validate(result.graph.model_dump())
        assert [n.op_type for n in graph.nodes] == [
            "Input",
            "Gemm",
            "Relu",
            "Gemm",
            "Softmax",
            "Output",
        ]
        assert [(e.source, e.target) for e in graph.edges] == [
            ("input_0", "node_0"),
            ("node_0", "node_1"),
            ("node_1", "node_2"),
            ("node_2", "node_3"),
            ("node_3", "output_0"),
        ]
        gemm = graph.nodes[1]
        assert gemm.attributes["transB"] == 1
        assert (
            gemm.parameter_summary
            and gemm.parameter_summary.parameter_count == 8 * 4 + 8
        )
        assert [i["shape"] for i in gemm.metadata["onnx"]["initializers"]] == [
            [8, 4],
            [8],
        ]
        assert result.report.parameter_count == 32 + 8 + 24 + 3

    def test_capabilities_are_inspection_only(self) -> None:
        caps = import_onnx(mlp_bytes(), "m", "mlp.onnx").graph.model.capabilities
        assert caps.graph is True
        assert not any(
            [caps.inference, caps.training, caps.activations, caps.gradients]
        )

    def test_dynamic_shapes_stay_symbolic_or_unknown(self) -> None:
        sample = build_sample("residual")
        assert sample is not None
        graph = import_onnx(sample.SerializeToString(), "m", "r.onnx").graph
        assert graph.nodes[0].output_shapes == [["N", 16, "H", "W"]]
        conv = graph.nodes[1]
        # Inference cannot know H and W after the conv; they are unknown, not invented.
        assert conv.output_shapes == [["N", 16, None, None]]

    def test_skip_connections_become_two_edges(self) -> None:
        sample = build_sample("residual")
        assert sample is not None
        graph = import_onnx(sample.SerializeToString(), "m", "r.onnx").graph
        add = next(n for n in graph.nodes if n.op_type == "Add")
        sources = sorted(e.source for e in graph.edges if e.target == add.id)
        assert sources == ["input_0", "node_2"]

    def test_unknown_operators_are_kept_with_a_warning(self) -> None:
        sample = build_sample("custom-op")
        assert sample is not None
        result = import_onnx(sample.SerializeToString(), "m", "c.onnx")
        custom = next(n for n in result.graph.nodes if n.op_type == "Scale")
        assert custom.unsupported is True
        assert custom.metadata["onnx"]["domain"] == "com.example"
        assert custom.attributes["factor"] == pytest.approx(2.0)
        assert [u.op_type for u in result.report.unsupported_operators] == ["Scale"]
        assert any("not a standard ONNX operator" in w for w in result.graph.warnings)
        # The graph around it is intact.
        assert any(e.target == custom.id for e in result.graph.edges)

    def test_subgraphs_are_reported_not_expanded(self) -> None:
        branch = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y_then"])],
            "then_branch",
            [],
            [helper.make_tensor_value_info("y_then", F32, [1])],
        )
        data = model_bytes(
            [
                helper.make_node(
                    "If", ["cond"], ["y"], then_branch=branch, else_branch=branch
                )
            ],
            [
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                helper.make_tensor_value_info("x", F32, [1]),
            ],
            [helper.make_tensor_value_info("y", F32, [1])],
        )
        result = import_onnx(data, "m", "if.onnx")
        node = next(n for n in result.graph.nodes if n.op_type == "If")
        assert node.attributes["then_branch"] == {
            "subgraphs": [{"name": "then_branch", "operators": 1}]
        }
        assert any("subgraph" in w for w in result.graph.warnings)

    def test_long_names_are_truncated(self) -> None:
        data = model_bytes(
            [helper.make_node("Relu", ["x"], ["y"], name="n" * 5000)],
            [helper.make_tensor_value_info("x", F32, [1])],
            [helper.make_tensor_value_info("y", F32, [1])],
        )
        node = import_onnx(data, "m", "long.onnx").graph.nodes[1]
        assert len(node.name) == 200 and node.name.endswith("…")


class TestUntrustedInput:
    @pytest.mark.parametrize(
        ("data", "message"),
        [
            (b"", "empty"),
            (
                b"\xff\xfe this is not protobuf \x00\x01\x02",
                "could not be parsed|no IR version",
            ),
            (b"\x08\x07", "no IR version or graph"),  # Valid protobuf, but no graph.
        ],
    )
    def test_malformed_files_are_rejected_safely(
        self, data: bytes, message: str
    ) -> None:
        with pytest.raises(OnnxImportError, match=message):
            import_onnx(data, "m", "bad.onnx")

    def test_too_many_operators(self) -> None:
        nodes = [
            helper.make_node("Relu", [f"t{i}"], [f"t{i + 1}"])
            for i in range(MAX_OPERATORS + 1)
        ]
        data = model_bytes(
            nodes,
            [helper.make_tensor_value_info("t0", F32, [1])],
            [helper.make_tensor_value_info(f"t{MAX_OPERATORS + 1}", F32, [1])],
        )
        with pytest.raises(OnnxImportError, match="at most 5,000"):
            import_onnx(data, "m", "big.onnx")

    def test_external_data_is_never_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        secret_dir = tempfile.mkdtemp()
        secret = os.path.join(secret_dir, "secret.bin")
        with open(secret, "wb") as f:
            f.write(np.ones(4, dtype=np.float32).tobytes())
        weight = TensorProto(
            name="w", data_type=F32, dims=[2, 2], data_location=TensorProto.EXTERNAL
        )
        for key, value in (("location", secret), ("offset", "0"), ("length", "16")):
            entry = weight.external_data.add()
            entry.key, entry.value = key, value
        data = model_bytes(
            [helper.make_node("MatMul", ["x", "w"], ["y"])],
            [helper.make_tensor_value_info("x", F32, [1, 2])],
            [helper.make_tensor_value_info("y", F32, [1, 2])],
            [weight],
        )

        opened: list[str] = []
        real_open = builtins.open
        monkeypatch.setattr(
            builtins,
            "open",
            lambda *a, **k: (opened.append(str(a[0])), real_open(*a, **k))[1],
        )
        monkeypatch.setattr(
            os,
            "stat",
            lambda *a, **k: pytest.fail("the importer touched the filesystem"),
        )
        result = import_onnx(data, "m", "ext.onnx")
        monkeypatch.undo()

        assert secret not in opened
        assert result.report.external_tensor_count == 1
        assert (
            result.graph.nodes[1].metadata["onnx"]["initializers"][0]["external"]
            is True
        )
        assert any("external files" in w for w in result.graph.warnings)


class TestImportApi:
    def test_upload_returns_graph_and_report(self, client: TestClient) -> None:
        response = upload(client, mlp_bytes(), "my_model.onnx")
        assert response.status_code == 201
        body = response.json()
        assert body["model_id"].startswith("onnx_")
        assert body["report"]["operator_count"] == 4
        assert body["report"]["file_name"] == "my_model.onnx"
        graph = client.get(f"/api/v1/models/{body['model_id']}/graph").json()
        assert graph == body["graph"]

    def test_wrong_extension(self, client: TestClient) -> None:
        response = upload(client, mlp_bytes(), "model.pt")
        assert response.status_code == 422
        assert response.json()["error"]["details"][0]["field"] == "query.filename"

    def test_parse_errors_are_safe(self, client: TestClient) -> None:
        response = upload(client, b"definitely not a model")
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "invalid_model"
        assert (
            "Traceback" not in response.text and "protobuf" not in response.text.lower()
        )

    def test_empty_upload(self, client: TestClient) -> None:
        assert upload(client, b"").status_code == 422

    def test_upload_size_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEUROVISION_MAX_UPLOAD_BYTES", "1024")
        get_settings.cache_clear()
        try:
            with TestClient(create_app()) as small:
                response = upload(small, b"\x00" * 2048)
                assert response.status_code == 413
                assert response.json()["error"]["code"] == "payload_too_large"
                assert (
                    upload(small, b"\x00" * 10).status_code == 422
                )  # Under the limit: parsed.
        finally:
            get_settings.cache_clear()

    @pytest.mark.parametrize("name", [s.name for s in SAMPLES])
    def test_samples_import(self, client: TestClient, name: str) -> None:
        response = client.post(f"/api/v1/import/onnx/samples/{name}")
        assert response.status_code == 201
        ModelGraph.model_validate(response.json()["graph"])

    def test_sample_list_and_unknown_sample(self, client: TestClient) -> None:
        names = [s["name"] for s in client.get("/api/v1/import/onnx/samples").json()]
        assert names == ["mlp", "cnn", "residual", "custom-op"]
        assert client.post("/api/v1/import/onnx/samples/nope").status_code == 404

    def test_imported_models_cannot_run_or_train(self, client: TestClient) -> None:
        model_id = upload(client, mlp_bytes()).json()["model_id"]
        base = f"/api/v1/models/{model_id}"
        for response in (
            client.post(f"{base}/forward", json={"inputs": [[0, 0, 0, 0]]}),
            client.post(f"{base}/training", json={}),
            client.get(f"{base}/metrics"),
        ):
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "capability_unavailable"


class TestTensors:
    def test_bounded_initializer_values(self, client: TestClient) -> None:
        body = upload(client, mlp_bytes()).json()
        tensor_id = body["graph"]["nodes"][1]["metadata"]["onnx"]["initializers"][0][
            "tensor_id"
        ]
        url = f"/api/v1/models/{body['model_id']}/tensors/{tensor_id}"
        full = client.get(url).json()
        assert (
            full["shape"] == [8, 4] and full["count"] == 32 and full["dtype"] == "float"
        )
        expected = numpy_helper.to_array(
            onnx.load_model_from_string(mlp_bytes()).graph.initializer[0]
        )
        assert full["values"] == pytest.approx(expected.reshape(-1).tolist())
        page = client.get(url, params={"offset": 30, "limit": 5}).json()
        assert len(page["values"]) == 2 and page["truncated"] is False
        assert client.get(url, params={"limit": 4097}).status_code == 422

    def test_unknown_tensor(self, client: TestClient) -> None:
        model_id = upload(client, mlp_bytes()).json()["model_id"]
        assert client.get(f"/api/v1/models/{model_id}/tensors/t99").status_code == 404

    def test_non_finite_values_are_reported_not_crashing(
        self, client: TestClient
    ) -> None:
        weights = numpy_helper.from_array(
            np.array([1.0, np.nan, np.inf, -2.0], dtype=np.float32), "w"
        )
        data = model_bytes(
            [helper.make_node("Mul", ["x", "w"], ["y"])],
            [helper.make_tensor_value_info("x", F32, [4])],
            [helper.make_tensor_value_info("y", F32, [4])],
            [weights],
        )
        model_id = upload(client, data).json()["model_id"]
        body = client.get(f"/api/v1/models/{model_id}/tensors/t0").json()
        assert body["values"] == [1.0, None, None, -2.0]
        assert body["non_finite_count"] == 2
        assert body["summary"]["min"] == -2.0 and body["summary"]["max"] == 1.0
