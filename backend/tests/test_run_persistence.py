"""External runs survive a backend restart (ADR 0013), and export metrics as CSV."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.runs import RunCreate, RunFinish
from app.services.external_runs import RESTARTED_MESSAGE, ExternalRunManager
from app.services.run_store import RunStore
from tests.test_runs_api import auth, create, log, step, stream


def finish(
    client: TestClient, run_id: str, token: str, status: str = "completed"
) -> None:
    response = client.post(
        f"/api/v1/runs/{run_id}/finish", json={"status": status}, headers=auth(token)
    )
    assert response.status_code == 200, response.text


def test_finished_run_is_saved_without_its_token(isolated_runs_dir: Path) -> None:
    with TestClient(create_app()) as client:
        run_id, token = create(client)
        assert log(client, run_id, token, [step(1), step(2)]).status_code == 202
        finish(client, run_id, token)

    saved = (isolated_runs_dir / f"{run_id}.json").read_text(encoding="utf-8")
    assert token not in saved
    document = json.loads(saved)
    assert document["schema_version"] == 1
    assert document["status"] == "completed"
    assert [e["event_type"] for e in document["events"]] == [
        "training.started",
        "training.step",
        "training.step",
        "training.completed",
    ]
    assert not list(isolated_runs_dir.glob("*.tmp"))


def test_reloaded_run_is_listed_replayable_and_read_only() -> None:
    with TestClient(create_app()) as client:
        run_id, token = create(client, name="before restart")
        log(client, run_id, token, [step(1), step(2), step(3)])
        finish(client, run_id, token)
        before = stream(client, run_id)
        info_before = client.get(f"/api/v1/runs/{run_id}").json()

    with TestClient(create_app()) as client:
        listed = client.get("/api/v1/runs").json()["runs"]
        assert [r["run_id"] for r in listed] == [run_id]
        assert listed[0] == {**info_before, "pending_commands": 0}
        assert stream(client, run_id) == before
        # The token is gone with the old process: the run cannot be written.
        refused = log(client, run_id, token, [step(4)])
        assert refused.status_code == 409
        assert "read-only" in refused.json()["error"]["message"]
        assert (
            client.post(
                f"/api/v1/runs/{run_id}/commands", json={"kind": "pause"}
            ).status_code
            == 409
        )


def test_run_active_at_shutdown_comes_back_failed() -> None:
    with TestClient(create_app()) as client:
        run_id, token = create(client)
        log(client, run_id, token, [step(1), step(2)])
    # Leaving the block ran the shutdown, which saved the active run.

    with TestClient(create_app()) as client:
        info = client.get(f"/api/v1/runs/{run_id}").json()
        assert info["status"] == "failed"
        assert info["steps_logged"] == 2
        events = stream(client, run_id)
        assert [e["step"] for e in events if e["event_type"] == "training.step"] == [
            1,
            2,
        ]
        assert events[-1]["event_type"] == "training.failed"
        assert events[-1]["message"] == RESTARTED_MESSAGE

    # The failure was saved, so a second restart does not add another one.
    with TestClient(create_app()) as client:
        events = stream(client, run_id)
        assert [e["event_type"] for e in events].count("training.failed") == 1


def test_run_left_running_by_a_crash_comes_back_failed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "crash")
    manager = ExternalRunManager(store=store)
    run, _ = manager.create(RunCreate(name="crashed", total_epochs=2))
    # No shutdown: only the save made when the run was created exists.

    reloaded = ExternalRunManager(store=store).get(run.run_id)
    assert reloaded is not None
    assert reloaded.status == "failed"
    assert reloaded.read_only
    assert not reloaded.token_matches("anything")


def test_reloaded_run_drops_the_link_to_its_in_memory_model(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "linked")
    manager = ExternalRunManager(store=store)
    run, _ = manager.create(RunCreate(name="linked", total_epochs=1, model_id="onnx_1"))
    run.finish(RunFinish(status="completed"), manager.now())
    assert run.info().model_id == "onnx_1"

    reloaded = ExternalRunManager(store=store).get(run.run_id)
    assert reloaded is not None
    assert reloaded.info().model_id is None


def test_retention_keeps_the_newest_finished_runs(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "kept")
    now = [1000.0]
    manager = ExternalRunManager(retained=2, store=store, clock=lambda: now[0])
    ids = []
    for i in range(3):
        now[0] += 1
        run, _ = manager.create(RunCreate(name=f"run {i}", total_epochs=1))
        run.finish(RunFinish(status="completed"), now[0])
        ids.append(run.run_id)
    assert sorted(p.stem for p in store.root.glob("run_*.json")) == sorted(ids[1:])

    # A smaller limit after a restart trims the oldest saved runs too.
    reloaded = ExternalRunManager(retained=1, store=store)
    assert [r.run_id for r in reloaded.list()] == [ids[2]]
    assert [p.stem for p in store.root.glob("run_*.json")] == [ids[2]]


def test_unreadable_saved_runs_are_skipped(isolated_runs_dir: Path) -> None:
    with TestClient(create_app()) as client:
        run_id, token = create(client)
        finish(client, run_id, token)
    (isolated_runs_dir / "run_broken.json").write_text("{not json", encoding="utf-8")
    document = json.loads((isolated_runs_dir / f"{run_id}.json").read_text("utf-8"))
    document["run_id"] = "run_tampered"
    document["events"].append({"event_type": "training.step", "step": "lots"})
    (isolated_runs_dir / "run_tampered.json").write_text(json.dumps(document), "utf-8")

    with TestClient(create_app()) as client:
        assert [r["run_id"] for r in client.get("/api/v1/runs").json()["runs"]] == [
            run_id
        ]


def test_metrics_csv_export() -> None:
    with TestClient(create_app()) as client:
        run_id, token = create(client)
        log(
            client,
            run_id,
            token,
            [
                {"epoch": 1, "step": 5, "metrics": {"loss": 0.5, "=cmd": 1.0}},
                {"epoch": 2, "step": 10, "metrics": {"loss": 0.25, "val_loss": 0.375}},
            ],
        )
        response = client.get(f"/api/v1/runs/{run_id}/metrics.csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert (
            f'filename="{run_id}-metrics.csv"'
            in response.headers["content-disposition"]
        )
        assert response.text.splitlines() == [
            "step,epoch,'=cmd,loss,val_loss",
            "5,1,1.0,0.5,",
            "10,2,,0.25,0.375",
        ]

        missing = client.get("/api/v1/runs/run_nope/metrics.csv")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "not_found"
