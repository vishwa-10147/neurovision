"""API tests: forward passes, tensors, training jobs, and the event stream."""

import json
import math
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import create_app
from app.schemas.events import training_event_adapter
from app.schemas.export import SHARED_SCHEMAS_DIR

EXAMPLE: dict[str, Any] = json.loads(
    (SHARED_SCHEMAS_DIR / "examples" / "dense-network.config.json").read_text(
        encoding="utf-8"
    )
)  # 2 -> 8 (relu) -> 6 (tanh) -> 2
TERMINAL = {"training.completed", "training.cancelled", "training.failed"}
SLOW = {"epochs": 500, "epoch_delay_ms": 50}


@pytest.fixture
def client() -> Iterator[TestClient]:
    # The context manager runs the lifespan, so jobs are cancelled on exit.
    with TestClient(create_app()) as test_client:
        yield test_client


def create(client: TestClient, **changes: Any) -> str:
    response = client.post("/api/v1/models/educational", json={**EXAMPLE, **changes})
    assert response.status_code == 201
    return str(response.json()["model_id"])


def start(client: TestClient, model_id: str, **config: Any) -> dict[str, Any]:
    response = client.post(f"/api/v1/models/{model_id}/training", json=config)
    assert response.status_code == 202, response.text
    return dict(response.json())


def stream(client: TestClient, job_id: str) -> list[dict[str, Any]]:
    """All events for a job, validated against the shared event schema."""
    events: list[dict[str, Any]] = []
    with client.websocket_connect(f"/api/v1/training/{job_id}/events") as ws:
        while True:
            event = ws.receive_json()
            training_event_adapter.validate_python(event)
            events.append(event)
            if event["event_type"] in TERMINAL:
                return events


class TestForward:
    def test_returns_real_layer_values(self, client: TestClient) -> None:
        model_id = create(client)
        body = client.post(
            f"/api/v1/models/{model_id}/forward",
            json={"inputs": [[0.5, -1.0], [1.5, 0.25]]},
        ).json()
        assert set(body["activations"]) == {"input", "hidden_1", "hidden_2", "output"}
        h1 = body["activations"]["hidden_1"]
        assert len(h1["post"]) == 2 and len(h1["post"][0]) == 8
        # ReLU: post == max(pre, 0), element by element.
        for pre_row, post_row in zip(h1["pre"], h1["post"], strict=True):
            assert post_row == pytest.approx([max(v, 0.0) for v in pre_row])
        h2 = body["activations"]["hidden_2"]
        assert h2["post"][0] == pytest.approx([math.tanh(v) for v in h2["pre"][0]])
        assert body["activations"]["input"]["pre"] is None
        for logits, prediction in zip(body["logits"], body["predictions"], strict=True):
            assert prediction == logits.index(max(logits))

    def test_capture_selects_layers(self, client: TestClient) -> None:
        model_id = create(client)
        body = client.post(
            f"/api/v1/models/{model_id}/forward",
            json={"inputs": [[0, 0]], "capture": ["hidden_2"]},
        ).json()
        assert list(body["activations"]) == ["hidden_2"]

    @pytest.mark.parametrize(
        ("payload", "field"),
        [
            ({"inputs": [[0, 0], [1, 2, 3]]}, "body.inputs.1"),
            ({"inputs": [[0, 0]], "capture": ["hidden_9"]}, "body.capture.0"),
            ({"inputs": []}, "body.inputs"),
            ({"inputs": [[0, 0]] * 65}, "body.inputs"),
            ({"inputs": [[0, 1e9]]}, "body.inputs.0.1"),
        ],
    )
    def test_invalid_inputs_name_the_field(
        self, client: TestClient, payload: dict[str, Any], field: str
    ) -> None:
        model_id = create(client)
        response = client.post(f"/api/v1/models/{model_id}/forward", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["details"][0]["field"] == field


class TestTensors:
    def test_weights_with_summary(self, client: TestClient) -> None:
        model_id = create(client)
        body = client.get(f"/api/v1/models/{model_id}/tensors/hidden_1.weight").json()
        assert body["shape"] == [8, 2]
        assert body["count"] == 16 and len(body["values"]) == 16
        assert body["truncated"] is False
        assert body["summary"]["min"] == pytest.approx(min(body["values"]))
        assert body["summary"]["max"] == pytest.approx(max(body["values"]))

    def test_slices_are_bounded(self, client: TestClient) -> None:
        model_id = create(client)
        url = f"/api/v1/models/{model_id}/tensors/hidden_2.weight"
        body = client.get(url, params={"offset": 40, "limit": 5}).json()
        assert body["offset"] == 40 and len(body["values"]) == 5
        assert body["truncated"] is True
        assert client.get(url, params={"limit": 5000}).status_code == 422
        assert client.get(url, params={"offset": 49}).status_code == 422

    def test_unknown_tensor_and_missing_gradients(self, client: TestClient) -> None:
        model_id = create(client)
        base = f"/api/v1/models/{model_id}/tensors"
        unknown = client.get(f"{base}/hidden_9.weight")
        assert unknown.status_code == 404
        no_grad = client.get(f"{base}/hidden_1.weight.grad")
        assert no_grad.status_code == 404
        assert "Train the model first" in no_grad.json()["error"]["message"]


class TestTraining:
    def test_streams_events_and_records_metrics(self, client: TestClient) -> None:
        model_id = create(client)
        job = start(client, model_id, epochs=8, capture_layers=["hidden_1"])
        assert job["status"] == "running" and job["total_epochs"] == 8

        events = stream(client, job["job_id"])
        kinds = [e["event_type"] for e in events]
        assert kinds == ["training.started"] + ["training.step"] * 8 + [
            "training.completed"
        ]
        steps = [e for e in events if e["event_type"] == "training.step"]
        assert [e["epoch"] for e in steps] == list(range(1, 9))
        assert set(steps[0]["metrics"]) == {
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
        }
        assert len(steps[-1]["selected_activations"]["hidden_1"]) == 8

        metrics = client.get(f"/api/v1/models/{model_id}/metrics").json()
        assert metrics["job"]["status"] == "completed"
        assert len(metrics["history"]) == 8
        grad = client.get(f"/api/v1/models/{model_id}/tensors/output.weight.grad")
        assert grad.status_code == 200 and grad.json()["shape"] == [2, 6]

    def test_late_subscribers_get_the_full_history(self, client: TestClient) -> None:
        model_id = create(client)
        job_id = start(client, model_id, epochs=3)["job_id"]
        first = stream(client, job_id)
        assert stream(client, job_id) == first

    def test_training_changes_the_weights(self, client: TestClient) -> None:
        model_id = create(client)
        url = f"/api/v1/models/{model_id}/tensors/hidden_1.weight"
        before = client.get(url).json()["values"]
        stream(client, start(client, model_id, epochs=3)["job_id"])
        assert client.get(url).json()["values"] != before

    def test_cancel_stops_the_job(self, client: TestClient) -> None:
        model_id = create(client)
        job_id = start(client, model_id, **SLOW)["job_id"]
        with client.websocket_connect(f"/api/v1/training/{job_id}/events") as ws:
            assert ws.receive_json()["event_type"] == "training.started"
            assert ws.receive_json()["event_type"] == "training.step"
            cancelled = client.post(f"/api/v1/training/{job_id}/cancel")
            assert cancelled.status_code == 202
            while (event := ws.receive_json())["event_type"] == "training.step":
                pass
            assert event["event_type"] == "training.cancelled"
        info = client.get(f"/api/v1/training/{job_id}").json()
        assert info["status"] == "cancelled"
        assert info["epoch"] < 500

    def test_one_job_per_model(self, client: TestClient) -> None:
        model_id = create(client)
        job_id = start(client, model_id, **SLOW)["job_id"]
        again = client.post(f"/api/v1/models/{model_id}/training", json={})
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "conflict"
        client.post(f"/api/v1/training/{job_id}/cancel")

    def test_concurrent_job_limit(self, client: TestClient) -> None:
        jobs = [start(client, create(client), **SLOW)["job_id"] for _ in range(2)]
        third = client.post(f"/api/v1/models/{create(client)}/training", json={})
        assert third.status_code == 429
        assert "2 training jobs" in third.json()["error"]["message"]
        for job_id in jobs:
            client.post(f"/api/v1/training/{job_id}/cancel")

    @pytest.mark.parametrize(
        ("model", "config", "field"),
        [
            ({"output_units": 3}, {"dataset": "moons"}, "body.dataset"),
            ({}, {"capture_layers": ["hidden_9"]}, "body.capture_layers.0"),
            ({}, {"learning_rate": 0}, "body.learning_rate"),
            ({}, {"epochs": 501}, "body.epochs"),
        ],
    )
    def test_invalid_training_requests(
        self,
        client: TestClient,
        model: dict[str, Any],
        config: dict[str, Any],
        field: str,
    ) -> None:
        model_id = create(client, **model)
        response = client.post(f"/api/v1/models/{model_id}/training", json=config)
        assert response.status_code == 422
        assert response.json()["error"]["details"][0]["field"] == field

    def test_spiral_supports_more_classes(self, client: TestClient) -> None:
        model_id = create(client, output_units=4)
        events = stream(
            client, start(client, model_id, dataset="spiral", epochs=2)["job_id"]
        )
        assert events[-1]["event_type"] == "training.completed"

    def test_unknown_job(self, client: TestClient) -> None:
        assert client.get("/api/v1/training/job_missing").status_code == 404
        assert client.post("/api/v1/training/job_missing/cancel").status_code == 404
        with client.websocket_connect("/api/v1/training/job_missing/events") as ws:
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
        assert closed.value.code == 4404

    def test_websocket_rejects_foreign_origins(self, client: TestClient) -> None:
        model_id = create(client)
        job_id = start(client, model_id, epochs=1)["job_id"]
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/api/v1/training/{job_id}/events",
                headers={"origin": "http://evil.example"},
            ) as ws:
                ws.receive_json()
