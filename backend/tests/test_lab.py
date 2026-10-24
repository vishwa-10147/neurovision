"""Neural Lab: only bundled scripts, only validated parameters, tracked processes."""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import lab as lab_service
from app.services.lab import TEMPLATES, LabError, LabManager, validate


def test_every_template_points_at_a_real_script() -> None:
    for script in TEMPLATES.values():
        assert (lab_service.EXAMPLES / script.file).is_file(), script.file


def test_validation_fills_defaults_and_checks_bounds() -> None:
    script, args = validate("cnn", {"blocks": 3, "lr": 0.01})
    assert script.file == "train_digits_cnn.py"
    assert (
        args[:2] == ["--blocks", "3"]
        and "--lr" in args
        and args[args.index("--lr") + 1] == "0.01"
    )
    assert args[args.index("--channels") + 1] == "16"  # Default.
    with pytest.raises(LabError, match="between 1 and 4"):
        validate("cnn", {"blocks": 9})
    with pytest.raises(LabError, match="whole number"):
        validate("cnn", {"blocks": 2.5})
    with pytest.raises(LabError, match="one of lstm, gru, rnn"):
        validate("rnn", {"cell": "transformer"})
    with pytest.raises(LabError, match="Unknown parameters"):
        validate("gan", {"shell": 1})
    with pytest.raises(LabError, match="divisible"):
        validate("transformer", {"dim": 30, "heads": 4})
    with pytest.raises(LabError, match="Unknown template"):
        validate("../../evil", {})


@pytest.fixture
def scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand-in scripts: print the arguments, or sleep, or fail."""
    (tmp_path / "train_spiral.py").write_text(
        "import sys\nprint('ARGS', sys.argv[1:])\n"
    )
    (tmp_path / "train_gan.py").write_text(
        "import time\nprint('started', flush=True)\ntime.sleep(30)\n"
    )
    (tmp_path / "train_gnn.py").write_text("raise SystemExit('boom')\n")
    monkeypatch.setattr(lab_service, "EXAMPLES", tmp_path)
    return tmp_path


def wait_for(manager: LabManager, job_id: str, status: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = next(j for j in manager.jobs() if j.job_id == job_id)
        if job.status == status:
            return job.model_dump()
        time.sleep(0.05)
    raise AssertionError(f"{job_id} never reached {status}")


def test_launches_the_script_with_validated_arguments(scripts: Path) -> None:
    manager = LabManager(python=sys.executable, max_jobs=2, enabled=True)
    job = manager.launch("mlp", {"epochs": 5}, "http://127.0.0.1:9")
    done = wait_for(manager, job.job_id, "finished")
    printed = " ".join(done["output"])
    assert "'--url', 'http://127.0.0.1:9'" in printed
    assert f"'--name', '{job.run_name}'" in printed and "'--epochs', '5'" in printed


def test_failures_are_reported_with_output(scripts: Path) -> None:
    manager = LabManager(python=sys.executable, max_jobs=2, enabled=True)
    job = manager.launch("gnn", {}, "http://127.0.0.1:9")
    failed = wait_for(manager, job.job_id, "failed")
    assert failed["exit_code"] == 1 and "boom" in " ".join(failed["output"])


def test_limits_concurrent_runs_and_stops_them(scripts: Path) -> None:
    manager = LabManager(python=sys.executable, max_jobs=1, enabled=True)
    job = manager.launch("gan", {}, "http://127.0.0.1:9")
    with pytest.raises(LabError, match="already training"):
        manager.launch("gan", {}, "http://127.0.0.1:9")
    assert manager.stop(job.job_id).status == "stopped"
    manager.launch("mlp", {}, "http://127.0.0.1:9")  # A slot is free again.
    manager.shutdown()


def test_disabled_lab_refuses_to_launch() -> None:
    with pytest.raises(LabError, match="turned off"):
        LabManager(python=None, max_jobs=1, enabled=False).launch("mlp", {}, "http://x")


def test_api_lists_templates_and_rejects_bad_requests() -> None:
    client = TestClient(create_app())
    listing = client.get("/api/v1/lab/templates").json()
    assert listing["enabled"] is True  # Development by default.
    ids = [t["id"] for t in listing["templates"]]
    assert ids == [
        "mlp",
        "cnn",
        "rnn",
        "transformer",
        "autoencoder",
        "vae",
        "gan",
        "gnn",
        "diffusion",
        "cartpole",
    ]
    response = client.post(
        "/api/v1/lab/runs", json={"template": "cnn", "params": {"blocks": 99}}
    )
    assert response.status_code == 422 and "between 1 and 4" in str(
        response.json()["error"]["details"]
    )
    assert client.post("/api/v1/lab/runs/lab_nope/stop").status_code == 404
    assert client.get("/api/v1/lab/runs").json() == {"jobs": []}
