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
        assert response.json() == {"accepted": 2, "steps_logged": 2, "commands": []}
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
        assert response.json() == {"accepted": 0, "steps_logged": 0, "commands": []}


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


# Health alerts and two-way control (ADR 0008) --------------------------------

ALL_CONTROLS = ["pause", "resume", "stop", "checkpoint", "set_lr"]


def command(client: TestClient, run_id: str, kind: str, **extra: Any) -> Any:
    return client.post(f"/api/v1/runs/{run_id}/commands", json={"kind": kind, **extra})


def ack(
    client: TestClient, run_id: str, token: str, command_id: int, **body: Any
) -> Any:
    return client.post(
        f"/api/v1/runs/{run_id}/commands/{command_id}/ack",
        json={"status": "applied", **body},
        headers=auth(token),
    )


def finish(
    client: TestClient, run_id: str, token: str, status: str = "completed"
) -> Any:
    return client.post(
        f"/api/v1/runs/{run_id}/finish", json={"status": status}, headers=auth(token)
    )


def of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [e for e in events if e["event_type"] == event_type]


class TestControls:
    def test_runs_are_not_controllable_by_default(self, client: TestClient) -> None:
        run_id, _ = create(client)
        info = client.get(f"/api/v1/runs/{run_id}").json()
        assert info["controls"] == [] and info["paused"] is False
        response = command(client, run_id, "pause")
        assert response.status_code == 409
        assert "control=True" in response.json()["error"]["message"]

    def test_command_lifecycle_is_streamed(self, client: TestClient) -> None:
        run_id, token = create(client, controls=ALL_CONTROLS)
        issued = command(client, run_id, "pause")
        assert issued.status_code == 202
        assert issued.json()["status"] == "pending"
        assert issued.json()["command_id"] == 1

        # The script sees it both by polling and on its next event batch.
        polled = client.get(f"/api/v1/runs/{run_id}/commands", headers=auth(token))
        assert [c["kind"] for c in polled.json()["commands"]] == ["pause"]
        logged = log(client, run_id, token, [step(1)])
        assert [c["command_id"] for c in logged.json()["commands"]] == [1]

        answered = ack(client, run_id, token, 1, paused=True, message="Paused.")
        assert answered.status_code == 200
        assert answered.json()["status"] == "applied"
        info = client.get(f"/api/v1/runs/{run_id}").json()
        assert info["paused"] is True and info["pending_commands"] == 0
        pending = client.get(f"/api/v1/runs/{run_id}/commands", headers=auth(token))
        assert pending.json() == {"commands": []}
        assert ack(client, run_id, token, 1).status_code == 409  # Answered once.

        finish(client, run_id, token)
        updates = of_type(stream(client, run_id), "run.command")
        assert [(u["command"]["status"], u["paused"]) for u in updates] == [
            ("pending", False),
            ("applied", True),
        ]
        assert updates[1]["command"]["message"] == "Paused."

    def test_set_lr_needs_a_valid_value(self, client: TestClient) -> None:
        run_id, token = create(client, controls=["pause", "resume", "stop", "set_lr"])
        assert command(client, run_id, "set_lr").status_code == 422
        assert command(client, run_id, "set_lr", value=-1).status_code == 422
        assert command(client, run_id, "set_lr", value=50).status_code == 422
        assert command(client, run_id, "pause", value=1).status_code == 422
        cid = command(client, run_id, "set_lr", value=0.001).json()["command_id"]
        ack(client, run_id, token, cid, learning_rate=0.001)
        assert client.get(f"/api/v1/runs/{run_id}").json()["learning_rate"] == 0.001

    def test_only_enabled_commands_are_accepted(self, client: TestClient) -> None:
        run_id, _ = create(client, controls=["pause", "resume", "stop"])
        assert command(client, run_id, "checkpoint").status_code == 409
        assert command(client, run_id, "stop").status_code == 202

    def test_pending_command_cap(self, client: TestClient) -> None:
        run_id, _ = create(client, controls=ALL_CONTROLS)
        for _ in range(16):
            assert command(client, run_id, "checkpoint").status_code == 202
        assert command(client, run_id, "checkpoint").status_code == 429

    def test_script_endpoints_need_the_token(self, client: TestClient) -> None:
        run_id, _ = create(client, controls=ALL_CONTROLS)
        cid = command(client, run_id, "pause").json()["command_id"]
        assert client.get(f"/api/v1/runs/{run_id}/commands").status_code == 401
        assert ack(client, run_id, "wrong-token", cid).status_code == 403

    def test_rejections_are_recorded(self, client: TestClient) -> None:
        run_id, token = create(client, controls=ALL_CONTROLS)
        cid = command(client, run_id, "resume").json()["command_id"]
        response = ack(
            client, run_id, token, cid, status="rejected", message="Not paused."
        )
        assert response.json()["status"] == "rejected"

    def test_pending_commands_expire_when_the_run_ends(
        self, client: TestClient
    ) -> None:
        run_id, token = create(client, controls=ALL_CONTROLS)
        command(client, run_id, "checkpoint")
        finish(client, run_id, token)
        updates = of_type(stream(client, run_id), "run.command")
        assert updates[-1]["command"]["status"] == "expired"
        assert command(client, run_id, "pause").status_code == 409

    def test_a_stopped_run_finishes_as_cancelled(self, client: TestClient) -> None:
        run_id, token = create(client, controls=ALL_CONTROLS)
        log(client, run_id, token, [step(1), step(2)])
        cid = command(client, run_id, "stop").json()["command_id"]
        ack(client, run_id, token, cid)
        assert (
            finish(client, run_id, token, "cancelled").json()["status"] == "cancelled"
        )
        assert stream(client, run_id)[-1] == {
            "schema_version": "1.0",
            "event_type": "training.cancelled",
            "job_id": run_id,
            "step": 2,
        }


def healthy(n: int, grad_norm: float = 1.0) -> dict[str, Any]:
    return step(
        n,
        health={
            "grad_norm": grad_norm,
            "learning_rate": 0.01,
            "parameters": {
                "fc1": {
                    "grad_norm": grad_norm,
                    "weight_norm": 3.0,
                    "update_ratio": 1e-3,
                    "parameters": 40,
                }
            },
            "activations": {
                "relu": {"kind": "relu", "mean": 0.2, "std": 0.3, "dead_fraction": 0.1}
            },
        },
    )


class TestHealth:
    def test_health_is_streamed_and_learning_rate_recorded(
        self, client: TestClient
    ) -> None:
        run_id, token = create(client)
        assert log(client, run_id, token, [healthy(1)]).status_code == 202
        assert client.get(f"/api/v1/runs/{run_id}").json()["learning_rate"] == 0.01
        finish(client, run_id, token)
        (logged,) = of_type(stream(client, run_id), "training.step")
        assert logged["health"]["parameters"]["fc1"]["update_ratio"] == 1e-3
        assert logged["health"]["activations"]["relu"]["dead_fraction"] == 0.1

    def test_exploding_gradients_raise_one_alert(self, client: TestClient) -> None:
        run_id, token = create(client)
        norms = [1.0, 1.0, 1.0, 1.0, 1.0, 80.0, 90.0]
        log(client, run_id, token, [healthy(i + 1, n) for i, n in enumerate(norms)])
        assert client.get(f"/api/v1/runs/{run_id}").json()["alerts"] == 1
        finish(client, run_id, token)
        (alert,) = of_type(stream(client, run_id), "run.alert")
        assert alert["alert"]["kind"] == "exploding_gradients"
        assert alert["alert"]["step"] == 6 and alert["alert"]["alert_id"] == 1

    def test_nan_loss_reported_by_name_is_critical(self, client: TestClient) -> None:
        run_id, token = create(client)
        log(client, run_id, token, [step(1, health={"nonfinite_metrics": ["loss"]})])
        finish(client, run_id, token, "failed")
        (alert,) = of_type(stream(client, run_id), "run.alert")
        assert alert["alert"]["severity"] == "critical"
        assert alert["alert"]["kind"] == "non_finite_loss"

    def test_rejects_out_of_range_health(self, client: TestClient) -> None:
        run_id, token = create(client)
        dead = {"kind": "relu", "mean": 0, "std": 1, "dead_fraction": 1.5}
        bad = step(1, health={"activations": {"r": dead}})
        assert log(client, run_id, token, [bad]).status_code == 422
        too_many = step(1, health={"parameters": {f"l{i}": {} for i in range(33)}})
        assert log(client, run_id, token, [too_many]).status_code == 422

    def test_history_keeps_layer_details_only_for_recent_steps(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(external_runs, "DETAILED_HEALTH_STEPS", 2)
        run_id, token = create(client)
        log(client, run_id, token, [healthy(i) for i in range(1, 5)])
        finish(client, run_id, token)
        steps = of_type(stream(client, run_id), "training.step")
        assert [bool(s["health"]["parameters"]) for s in steps] == [
            False,
            False,
            True,
            True,
        ]
        # The global numbers stay for trend charts.
        assert all(s["health"]["grad_norm"] == 1.0 for s in steps)
