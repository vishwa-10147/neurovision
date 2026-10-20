"""2D views of run tensors: feature maps, attention heads, and their bounds."""

import io
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.ml.tensor_maps import MapRequestError, downsample, tensor_grid, tensor_map


class TestMapMath:
    def test_downsample_averages_blocks(self) -> None:
        matrix = np.arange(16, dtype=float).reshape(4, 4)
        small, stride = downsample(matrix, max_cells=4)
        assert stride == 2
        assert small.tolist() == [[2.5, 4.5], [10.5, 12.5]]

    def test_small_maps_are_exact(self) -> None:
        matrix = np.arange(6, dtype=float).reshape(2, 3)
        small, stride = downsample(matrix, max_cells=100)
        assert stride == 1 and small.tolist() == matrix.tolist()

    def test_non_finite_values_are_ignored_in_blocks(self) -> None:
        matrix = np.array([[1.0, np.nan], [3.0, np.inf]])
        small, _ = downsample(matrix, max_cells=1)
        assert small.tolist() == [[2.0]]

    def test_uneven_sizes_never_exceed_the_budget(self) -> None:
        matrix = np.ones((300, 257))
        small, stride = downsample(matrix, max_cells=65_536 // 4)
        assert small.size <= 65_536 // 4 and stride > 1

    def test_map_selects_by_leading_indices(self) -> None:
        tensor = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
        view = tensor_map(tensor, [1, 2], max_cells=1000)
        assert (view["rows"], view["cols"], view["stride"]) == (4, 5, 1)
        assert view["values"] == tensor[1, 2].tolist()
        assert (view["min"], view["max"]) == (tensor[1, 2].min(), tensor[1, 2].max())

    def test_grid_covers_the_axis_up_to_a_cap(self) -> None:
        tensor = np.zeros((1, 10, 8, 8))
        grid = tensor_grid(tensor, [0], max_items=4, thumb_cells=16)
        assert grid["items_total"] == 10 and len(grid["items"]) == 4
        assert grid["items"][0]["stride"] == 2 and len(grid["items"][0]["values"]) == 4

    @pytest.mark.parametrize(
        ("shape", "fixed", "message"),
        [
            ((3, 4), [0], "Give 0 leading"),
            ((2, 3, 4), [5], "out of range"),
            ((4,), [], "at least 2"),
        ],
    )
    def test_map_requests_must_fit_the_tensor(
        self, shape: tuple[int, ...], fixed: list[int], message: str
    ) -> None:
        with pytest.raises(MapRequestError, match=message):
            tensor_map(np.zeros(shape), fixed, max_cells=100)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def sample_run(
    client: TestClient, name: str, inputs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    imported = client.post(f"/api/v1/import/onnx/samples/{name}").json()
    run = client.post(
        f"/api/v1/models/{imported['model_id']}/run", json={"inputs": inputs}
    )
    assert run.status_code == 200, run.text
    return imported, run.json()


def view(client: TestClient, kind: str, model_id: str, **params: Any) -> Any:
    return client.get(f"/api/v1/models/{model_id}/{kind}", params=params)


class TestFeatureMaps:
    def test_channel_grid_and_detail_use_the_run_values(
        self, client: TestClient
    ) -> None:
        imported, run = sample_run(
            client, "cnn", {"image": {"shape": [1, 1, 28, 28], "seed": 3}}
        )
        model_id, run_id = imported["model_id"], run["run_id"]
        assert run["retained"] is True

        grid = view(
            client, "grids", model_id, run_id=run_id, node_id="node_0", fixed=[0]
        ).json()
        assert grid["shape"] == [1, 8, 28, 28] and grid["axis"] == 1
        assert grid["items_total"] == 8 and (grid["rows"], grid["cols"]) == (28, 28)

        detail = view(
            client, "maps", model_id, run_id=run_id, node_id="node_0", fixed=[0, 0]
        ).json()
        assert detail["stride"] == 1 and len(detail["values"]) == 28
        # Channel 0 is the first 784 row-major values of the run's conv output.
        conv = next(t for t in run["intermediates"] if t["node_id"] == "node_0")
        assert np.ravel(detail["values"]) == pytest.approx(
            conv["values"][:784], abs=1e-6
        )
        # Each 24×24-budget thumbnail is the 2×2 block mean of the channel.
        thumb = np.array(grid["items"][0]["values"])
        expected = np.array(detail["values"]).reshape(14, 2, 14, 2).mean(axis=(1, 3))
        assert thumb == pytest.approx(expected, abs=1e-6)

    def test_inputs_can_be_viewed(self, client: TestClient) -> None:
        imported, run = sample_run(
            client, "cnn", {"image": {"shape": [2, 1, 28, 28], "fill": "range"}}
        )
        image = view(
            client,
            "maps",
            imported["model_id"],
            run_id=run["run_id"],
            node_id="input_0",
            fixed=[1, 0],
        ).json()
        assert image["tensor"] == "image" and image["rows"] == 28
        assert image["min"] >= -1 and image["max"] <= 1
        assert run["inputs"] == [
            {
                "name": "image",
                "node_id": "input_0",
                "shape": [2, 1, 28, 28],
                "dtype": "float32",
            }
        ]

    @pytest.mark.parametrize(
        ("kind", "params", "field"),
        [
            ("maps", {"node_id": "node_0", "fixed": [0]}, "query.fixed"),
            ("maps", {"node_id": "node_0", "fixed": [0, 99]}, "query.fixed"),
            (
                "grids",
                {"node_id": "node_7", "fixed": [0]},
                "query.fixed",
            ),  # logits [1, 10]
            ("maps", {"node_id": "node_99", "fixed": [0, 0]}, "query.node_id"),
            (
                "maps",
                {"node_id": "node_0", "output": 3, "fixed": [0, 0]},
                "query.output",
            ),
        ],
    )
    def test_invalid_views_name_the_field(
        self, client: TestClient, kind: str, params: dict[str, Any], field: str
    ) -> None:
        imported, run = sample_run(client, "cnn", {"image": {"shape": [1, 1, 28, 28]}})
        response = view(
            client, kind, imported["model_id"], run_id=run["run_id"], **params
        )
        assert response.status_code == 422
        assert response.json()["error"]["details"][0]["field"] == field

    def test_expired_runs_say_to_run_again(self, client: TestClient) -> None:
        imported, run = sample_run(client, "cnn", {"image": {"shape": [1, 1, 28, 28]}})
        unknown = view(
            client,
            "maps",
            imported["model_id"],
            run_id="run_missing",
            node_id="node_0",
            fixed=[0, 0],
        )
        assert (
            unknown.status_code == 410
            and unknown.json()["error"]["code"] == "run_expired"
        )
        runtime = client.app.state.onnx_runtime  # type: ignore[attr-defined]
        runtime._process.kill()
        runtime._process.join(5)
        gone = view(
            client,
            "maps",
            imported["model_id"],
            run_id=run["run_id"],
            node_id="node_0",
            fixed=[0, 0],
        )
        assert (
            gone.status_code == 410
            and "Run the model again" in gone.json()["error"]["message"]
        )

    def test_educational_models_have_no_map_views(self, client: TestClient) -> None:
        config = {"name": "n", "input_units": 2, "hidden_layers": [], "output_units": 2}
        model_id = client.post("/api/v1/models/educational", json=config).json()[
            "model_id"
        ]
        response = view(
            client, "maps", model_id, run_id="run_x", node_id="output", fixed=[]
        )
        assert response.status_code == 400


class TestAttention:
    def test_sample_attention_is_detected_and_rows_sum_to_one(
        self, client: TestClient
    ) -> None:
        ids = [3, 14, 15, 9, 26, 5, 35]
        imported, run = sample_run(
            client, "attention", {"ids": {"shape": [1, 7], "values": ids}}
        )
        assert imported["graph"]["model"]["capabilities"]["attention"] is True
        [candidate] = imported["report"]["attention"]
        assert candidate["shape"] == ["batch", 4, "seq", "seq"]
        assert candidate["layout"] == ["batch", "heads", "query", "key"]
        assert (candidate["name"], candidate["tensor"]) == (
            "attention_weights",
            "weights",
        )

        heads = view(
            client,
            "grids",
            imported["model_id"],
            run_id=run["run_id"],
            node_id=candidate["node_id"],
            fixed=[0],
        ).json()
        assert heads["items_total"] == 4 and (heads["rows"], heads["cols"]) == (7, 7)
        for head in range(4):
            matrix = np.array(
                view(
                    client,
                    "maps",
                    imported["model_id"],
                    run_id=run["run_id"],
                    node_id=candidate["node_id"],
                    fixed=[0, head],
                ).json()["values"]
            )
            assert matrix.sum(axis=1) == pytest.approx(np.ones(7), abs=1e-5)
            assert matrix.min() >= 0

    def test_models_without_square_softmax_report_no_attention(
        self, client: TestClient
    ) -> None:
        imported = client.post("/api/v1/import/onnx/samples/mlp").json()
        assert imported["report"]["attention"] == []
        assert imported["graph"]["model"]["capabilities"]["attention"] is False

    def test_a_real_pytorch_multihead_attention_matches_torch(
        self, client: TestClient
    ) -> None:
        torch = pytest.importorskip("torch")
        torch.manual_seed(0)
        mha = torch.nn.MultiheadAttention(
            embed_dim=16, num_heads=4, batch_first=True
        ).eval()

        class SelfAttention(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mha = mha

            def forward(self, x: Any) -> Any:
                return self.mha(x, x, x, need_weights=False)[0]

        buffer = io.BytesIO()
        torch.onnx.export(
            SelfAttention(),
            torch.zeros(1, 6, 16),
            buffer,
            input_names=["x"],
            dynamo=False,
        )
        imported = client.post(
            "/api/v1/import/onnx",
            params={"filename": "mha.onnx"},
            content=buffer.getvalue(),
        ).json()
        [candidate] = imported["report"]["attention"]
        assert candidate["shape"] == [1, 4, 6, 6]

        x = torch.randn(1, 6, 16)
        run = client.post(
            f"/api/v1/models/{imported['model_id']}/run",
            json={
                "inputs": {"x": {"shape": [1, 6, 16], "values": x.reshape(-1).tolist()}}
            },
        ).json()
        with torch.no_grad():
            _, expected = mha(x, x, x, need_weights=True, average_attn_weights=False)
        for head in range(4):
            got = view(
                client,
                "maps",
                imported["model_id"],
                run_id=run["run_id"],
                node_id=candidate["node_id"],
                fixed=[0, head],
            ).json()
            assert np.array(got["values"]) == pytest.approx(
                expected[0, head].numpy(), abs=1e-5
            )
