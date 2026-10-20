"""External runs: creation, write tokens, bounds, ordering, and streaming."""

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.events import training_event_adapter
from app.services import external_runs
from app.services.external_runs import IDLE_TIMEOUT_S, ExternalRunManager

TERMINAL = {"training.completed", "training.cancelled", "training.failed"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def create(client: TestClient, **changes: Any) -> tuple[str, str]:
    body = {"name": "my experiment", "total_epochs": 3, **changes}
    response = client.post("/api/v1/runs", json=body)
    assert response.status_code == 201, response.text
    data = response.json()
    return data["run"]["run_id"], data["write_token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def step(n: int, epoch: int | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "epoch": n if epoch is None else epoch,
        "step": n,
        "metrics": {"loss": 1 / n, "accuracy": 0.5 + n / 100},
        **extra,
    }


def log(client: TestClient, run_id: str, token: str, events: list[Any]) -> Any:
    return client.post(
        f"/api/v1/runs/{run_id}/events", json={"events": events}, headers=auth(token)
    )


def stream(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with client.websocket_connect(f"/api/v1/training/{run_id}/events") as ws:
        while True:
            event = ws.receive_json()
            training_event_adapter.validate_python(event)
            events.append(event)
            if event["event_type"] in TERMINAL:
                return events


def stream_first(client: TestClient, run_id: str) -> dict[str, Any]:
    with client.websocket_connect(f"/api/v1/training/{run_id}/events") as ws:
        return dict(ws.receive_json())


def error_code(response: Any) -> str:
    return str(response.json()["error"]["code"])


class TestLifecycle:
    def test_create_log_finish_and_replay(self, client: TestClient) -> None:
        run_id, token = create(client)
        assert run_id.startswith("run_") and len(token) >= 32

        response = log(client, run_id, token, [step(1), step(2)])
        assert response.status_code == 202
        assert response.json() == {"accepted": 2, "steps_logged": 2}
        finished = client.post(
            f"/api/v1/runs/{run_id}/finish",
            json={"status": "completed"},
            headers=auth(token),
        )
        assert finished.status_code == 200
        assert finished.json()["status"] == "completed"

        events = stream(client, run_id)
        assert [e["event_type"] for e in events] == [
            "training.started",
            "training.step",
            "training.step",
            "training.completed",
        ]
        assert events[0]["total_epochs"] == 3
        assert events[0]["model_id"] == run_id
        assert events[2]["metrics"] == {"loss": 0.5, "accuracy": 0.52}
        # Completion repeats the last logged metrics, not invented ones.
        assert events[3]["metrics"] == events[2]["metrics"]
        assert events[3]["step"] == 2

    def test_websocket_follows_live_steps(self, client: TestClient) -> None:
        run_id, token = create(client)
        with client.websocket_connect(f"/api/v1/training/{run_id}/events") as ws:
            assert ws.receive_json()["event_type"] == "training.started"
            assert log(client, run_id, token, [step(1)]).status_code == 202
            live = ws.receive_json()
            assert live["event_type"] == "training.step" and live["step"] == 1
            client.post(
                f"/api/v1/runs/{run_id}/finish",
                json={"status": "failed", "message": "Out of memory on GPU 0."},
                headers=auth(token),
            )
            failed = ws.receive_json()
            assert failed == {
                "schema_version": "1.0",
                "event_type": "training.failed",
                "job_id": run_id,
                "message": "Out of memory on GPU 0.",
            }

    def test_list_and_get_never_expose_the_token(self, client: TestClient) -> None:
        run_id, token = create(client, name="first")
        create(client, name="second")
        log(client, run_id, token, [step(1, epoch=1)])
        listed = client.get("/api/v1/runs")
        assert listed.status_code == 200
        runs = listed.json()["runs"]
        assert [r["name"] for r in runs] == ["second", "first"]  # newest first
        assert token not in listed.text
        info = client.get(f"/api/v1/runs/{run_id}").json()
        assert info["steps_logged"] == 1 and info["epoch"] == 1 and info["step"] == 1
        assert "write_token" not in info

    def test_links_an_existing_model(self, client: TestClient) -> None:
        sample = client.post("/api/v1/import/onnx/samples/mlp")
        assert sample.status_code == 201
        model_id = sample.json()["model_id"]
        run_id, _ = create(client, model_id=model_id)
        assert client.get(f"/api/v1/runs/{run_id}").json()["model_id"] == model_id
        started = stream_first(client, run_id)
        assert started["model_id"] == model_id

    def test_rejects_an_unknown_model_link(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/runs",
            json={"name": "x", "total_epochs": 1, "model_id": "onnx_missing"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["details"][0]["field"] == "body.model_id"

    def test_empty_batch_is_a_heartbeat(self, client: TestClient) -> None:
        run_id, token = create(client)
        response = log(client, run_id, token, [])
        assert response.status_code == 202
        assert response.json() == {"accepted": 0, "steps_logged": 0}


class TestWriteToken:
    def test_missing_token(self, client: TestClient) -> None:
        run_id, _ = create(client)
        response = client.post(f"/api/v1/runs/{run_id}/events", json={"events": []})
        assert response.status_code == 401
        assert error_code(response) == "unauthorized"

    def test_wrong_token_and_other_runs_token(self, client: TestClient) -> None:
        run_id, _ = create(client)
        _, other_token = create(client)
        for token in ("not-the-token", other_token):
            response = log(client, run_id, token, [step(1)])
            assert response.status_code == 403
            assert error_code(response) == "forbidden"
            finish = client.post(
                f"/api/v1/runs/{run_id}/finish",
                json={"status": "completed"},
                headers=auth(token),
            )
            assert finish.status_code == 403
        assert client.get(f"/api/v1/runs/{run_id}").json()["steps_logged"] == 0

    def test_unknown_run(self, client: TestClient) -> None:
        response = log(client, "run_missing", "token", [step(1)])
        assert response.status_code == 404


class TestBounds:
    @pytest.mark.parametrize(
        ("event", "field"),
        [
            (step(1, metrics={f"m{i}": 0.0 for i in range(17)}), "metrics"),
            (
                step(1, selected_activations={f"l{i}": [0.0] for i in range(9)}),
                "selected_activations",
            ),
            (step(1, selected_activations={"fc1": [0.0] * 257}), "fc1"),
            (step(1, unexpected=True), "unexpected"),
            ({**step(1), "step": -1}, "step"),
        ],
    )
    def test_rejects_oversized_or_malformed_steps(
        self, client: TestClient, event: dict[str, Any], field: str
    ) -> None:
        run_id, token = create(client)
        response = log(client, run_id, token, [event])
        assert response.status_code == 422
        fields = [d["field"] for d in response.json()["error"]["details"]]
        assert any(f.startswith("body.events.0") and field in f for f in fields)

    def test_accepts_the_largest_allowed_step(self, client: TestClient) -> None:
        run_id, token = create(client)
        event = step(
            1,
            metrics={f"m{i}": float(i) for i in range(16)},
            selected_activations={f"l{i}": [0.5] * 256 for i in range(8)},
        )
        assert log(client, run_id, token, [event] * 1).status_code == 202

    def test_rejects_too_many_events_per_request(self, client: TestClient) -> None:
        run_id, token = create(client)
        response = log(client, run_id, token, [step(n) for n in range(1, 34)])
        assert response.status_code == 422

    def test_rejects_non_finite_values(self, client: TestClient) -> None:
        run_id, token = create(client)
        body = '{"events": [{"epoch": 1, "step": 1, "metrics": {"loss": NaN}}]}'
        response = client.post(
            f"/api/v1/runs/{run_id}/events",
            content=body,
            headers={**auth(token), "Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "NaN" not in response.json()["error"]["message"]

    def test_rejects_invalid_json_safely(self, client: TestClient) -> None:
        run_id, token = create(client)
        response = client.post(
            f"/api/v1/runs/{run_id}/events", content=b"{not json", headers=auth(token)
        )
        assert response.status_code == 422
        assert error_code(response) == "validation_error"

    def test_rejects_oversized_bodies(self, client: TestClient) -> None:
        run_id, token = create(client)
        response = client.post(
            f"/api/v1/runs/{run_id}/events",
            content=b" " * (2 * 1024 * 1024 + 1),
            headers=auth(token),
        )
        assert response.status_code == 413
        assert error_code(response) == "payload_too_large"

    def test_steps_must_increase_and_batches_are_all_or_nothing(
        self, client: TestClient
    ) -> None:
        run_id, token = create(client)
        assert log(client, run_id, token, [step(1)]).status_code == 202
        response = log(client, run_id, token, [step(2), step(2)])
        assert response.status_code == 422
        assert response.json()["error"]["details"][0]["field"] == "body.events.1.step"
        response = log(client, run_id, token, [step(3, epoch=0)])
        assert response.status_code == 422
        assert client.get(f"/api/v1/runs/{run_id}").json()["steps_logged"] == 1

    def test_per_run_step_cap(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(external_runs, "MAX_STEPS_PER_RUN", 3)
        run_id, token = create(client)
        assert log(client, run_id, token, [step(1), step(2)]).status_code == 202
        response = log(client, run_id, token, [step(3), step(4)])
        assert response.status_code == 409
        assert "limit of 3 logged steps" in response.json()["error"]["message"]
        assert log(client, run_id, token, [step(3)]).status_code == 202

    def test_finished_runs_reject_more_events(self, client: TestClient) -> None:
        run_id, token = create(client)
        client.post(
            f"/api/v1/runs/{run_id}/finish",
            json={"status": "completed"},
            headers=auth(token),
        )
        response = log(client, run_id, token, [step(1)])
        assert response.status_code == 409
        assert "already completed" in response.json()["error"]["message"]

    def test_active_run_cap(self, client: TestClient) -> None:
        client.app.state.external_runs = ExternalRunManager(max_active=1)  # type: ignore[attr-defined]
        create(client)
        response = client.post("/api/v1/runs", json={"name": "x", "total_epochs": 1})
        assert response.status_code == 429

    def test_history_keeps_only_the_latest_activations(
        self, client: TestClient
    ) -> None:
        run_id, token = create(client)
        values = {"fc1": [0.1, 0.2]}
        log(
            client,
            run_id,
            token,
            [
                step(1, selected_activations=values),
                step(2),
                step(3, selected_activations={"fc1": [0.3, 0.4]}),
            ],
        )
        client.post(
            f"/api/v1/runs/{run_id}/finish",
            json={"status": "completed"},
            headers=auth(token),
        )
        steps = [
            e for e in stream(client, run_id) if e["event_type"] == "training.step"
        ]
        assert [s["selected_activations"] for s in steps] == [
            {},
            {},
            {"fc1": [0.3, 0.4]},
        ]
        # Metrics stay on every step.
        assert [s["metrics"]["loss"] for s in steps] == [1.0, 0.5, pytest.approx(1 / 3)]


class TestIdleRuns:
    def test_quiet_runs_are_closed_with_an_explanation(
        self, client: TestClient
    ) -> None:
        now = [1000.0]
        client.app.state.external_runs = ExternalRunManager(clock=lambda: now[0])  # type: ignore[attr-defined]
        run_id, token = create(client)
        now[0] += IDLE_TIMEOUT_S - 1
        assert log(client, run_id, token, []).status_code == 202  # heartbeat
        now[0] += IDLE_TIMEOUT_S - 1
        assert client.get(f"/api/v1/runs/{run_id}").json()["status"] == "running"
        now[0] += 2
        assert client.get(f"/api/v1/runs/{run_id}").json()["status"] == "failed"
        events = stream(client, run_id)
        assert "run.finish()" in events[-1]["message"]
        assert log(client, run_id, token, [step(1)]).status_code == 409


def test_shared_schema_example_is_valid() -> None:
    from app.schemas.export import SHARED_SCHEMAS_DIR
    from app.schemas.runs import EventBatch

    path = SHARED_SCHEMAS_DIR / "examples" / "external-run.events.json"
    EventBatch.model_validate(json.loads(path.read_text(encoding="utf-8")))
