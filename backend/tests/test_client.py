"""The Python client in sdk/neurovision_client.py: batching, sanitizing,
never raising into the training loop, activation probes, and end to end."""

import json
import math
import socket
import threading
import time
import urllib.request
from collections.abc import Iterator
from typing import Any

import neurovision_client as nv
import pytest
import torch
import uvicorn
from torch import nn

from app.main import create_app


class FakeServer:
    """A transport that records requests and answers like the backend."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Any, dict[str, str]]] = []
        self.event_status = 202
        self.fail_network = False
        self.delay = 0.0

    def __call__(
        self, method: str, url: str, body: Any, headers: dict[str, str]
    ) -> tuple[int, Any]:
        if self.fail_network:
            raise ConnectionRefusedError("connection refused")
        # Recorded when sent, so the order shows what was sent after what.
        self.requests.append((method, url, body, headers))
        time.sleep(self.delay)
        if url.endswith("/api/v1/runs"):
            return 201, {"run": {"run_id": "run_abc"}, "write_token": "secret-token"}
        if url.endswith("/events"):
            return self.event_status, {"error": {"message": "nope"}}
        return 200, {}

    def events(self) -> list[dict[str, Any]]:
        return [
            e
            for _, url, body, _ in self.requests
            if url.endswith("/events")
            for e in body["events"]
        ]

    def finish_body(self) -> Any:
        return next(b for _, url, b, _ in self.requests if url.endswith("/finish"))


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


def start(server: FakeServer, **options: Any) -> nv.Run:
    return nv.start_run(
        "test", total_epochs=5, url="http://nv.test", transport=server, **options
    )


def test_creates_logs_and_finishes(server: FakeServer) -> None:
    run = start(server, track_system=False)
    assert run.enabled and run.run_id == "run_abc"
    assert server.requests[0][2] == {"name": "test", "total_epochs": 5}
    for epoch in range(1, 4):
        run.log(epoch=epoch, metrics={"loss": 1 / epoch, "acc": torch.tensor(0.5)})
    run.finish()
    events = server.events()
    assert [e["step"] for e in events] == [0, 1, 2]
    assert events[1] == {
        "epoch": 2,
        "step": 1,
        "metrics": {"loss": 0.5, "acc": 0.5},
        "selected_activations": {},
    }
    assert server.finish_body() == {"status": "completed"}
    # Every write carries the token; it is never put in a URL.
    for _, url, _, headers in server.requests[1:]:
        assert headers["Authorization"] == "Bearer secret-token"
        assert "secret-token" not in url
    assert run.sent == 3 and run.dropped == 0


def test_batches_at_most_32_steps_per_request(server: FakeServer) -> None:
    run = start(server, flush_interval=10)
    for i in range(70):
        run.log(epoch=0, metrics={"loss": float(i)})
    run.finish()
    sizes = [
        len(b["events"]) for _, url, b, _ in server.requests if url.endswith("/events")
    ]
    assert sum(sizes) == 70 and max(sizes) <= 32


def test_sanitizes_values_the_server_would_reject(server: FakeServer) -> None:
    run = start(server)
    run.log(
        epoch=1,
        step=10,
        metrics={"loss": float("nan"), **{f"m{i}": i for i in range(20)}},
        activations={
            **{f"layer{i}": [1.0, 2.0] for i in range(10)},
        },
    )
    run.log(epoch=1, step=10, metrics={"loss": 1.0})  # step did not increase
    run.log(
        epoch=2,
        step=11,
        metrics={},
        activations={"big": torch.arange(1000.0), "bad": [math.inf], "np": [[1, 2]]},
    )
    run.finish()
    first, second = server.events()
    assert "loss" not in first["metrics"] and len(first["metrics"]) == 16
    assert len(first["selected_activations"]) == 8
    assert second["step"] == 11
    assert second["selected_activations"]["big"] == [float(i) for i in range(256)]
    assert "bad" not in second["selected_activations"]
    assert second["selected_activations"]["np"] == [1.0, 2.0]


def test_unreachable_server_gives_a_disabled_run(server: FakeServer) -> None:
    server.fail_network = True
    run = start(server)
    assert not run.enabled
    run.log(epoch=1, metrics={"loss": 1.0})  # no-op, no exception
    run.finish()


def test_network_errors_never_raise_into_the_loop(server: FakeServer) -> None:
    run = start(server)
    server.fail_network = True
    for i in range(5):
        run.log(epoch=i, metrics={"loss": 1.0})
    run.flush()
    run.finish()  # finish also fails to send; still no exception
    assert run.sent == 0


def test_unreadable_values_are_skipped(server: FakeServer) -> None:
    run = start(server)
    run.log(epoch=1, metrics={"loss": "not a number"})
    run.log(epoch=1, metrics={"loss": 1.0})
    run.finish()
    assert [e["metrics"] for e in server.events()] == [{"loss": 1.0}]


def test_stops_sending_when_the_run_is_closed(server: FakeServer) -> None:
    run = start(server)
    server.event_status = 409
    run.log(epoch=1, metrics={"loss": 1.0})
    run.flush()
    sent_before = len(server.requests)
    run.log(epoch=2, metrics={"loss": 1.0})
    run.flush()
    assert len(server.requests) == sent_before
    assert not run.enabled


def test_drops_oldest_steps_under_backpressure(server: FakeServer) -> None:
    server.delay = 0.2
    run = start(server, max_queue=4, flush_interval=10)
    started = time.perf_counter()
    for i in range(50):
        run.log(epoch=0, metrics={"loss": float(i)})
    # Logging never waits for the network.
    assert time.perf_counter() - started < 0.1
    run.finish(timeout=10)
    assert run.dropped > 0
    losses = [e["metrics"]["loss"] for e in server.events()]
    assert losses[-1] == 49.0  # the newest steps survive


def test_finish_accounts_for_every_step_on_a_slow_server(server: FakeServer) -> None:
    server.delay = 0.3
    run = start(server, flush_interval=0.05)
    for i in range(100):
        run.log(epoch=0, metrics={"loss": float(i)})
    run.finish(timeout=0.2)
    time.sleep(0.5)  # let a request that was in flight complete
    assert run.sent + run.dropped == 100
    assert run.dropped > 0
    # Nothing is sent after the run is finished.
    assert server.requests[-1][1].endswith("/finish")


def test_context_manager_reports_failures_without_details(server: FakeServer) -> None:
    with pytest.raises(RuntimeError):
        with start(server) as run:
            run.log(epoch=1, metrics={"loss": 1.0})
            raise RuntimeError("C:/secret/path/data.csv is corrupt")
    assert server.finish_body() == {
        "status": "failed",
        "message": "The training script raised RuntimeError.",
    }


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.fc(x)))


def test_activation_probe_measures_post_activation_values() -> None:
    torch.manual_seed(0)
    model = Net()
    probe_batch = torch.randn(2, 3)
    probe = nv.ActivationProbe(model, {"hidden": model.act}, probe_batch)
    values = probe.capture()
    expected = torch.relu(model.fc(probe_batch[0])).tolist()
    assert values["hidden"] == pytest.approx(expected)
    # Training mode is restored and hooks are removed afterwards.
    assert model.training
    assert not model.act._forward_hooks
    with pytest.raises(ValueError):
        nv.ActivationProbe(model, {}, probe_batch)


def test_activation_probe_does_not_touch_gradients() -> None:
    model = Net()
    probe = nv.ActivationProbe(model, {"fc": model.fc}, torch.randn(1, 3))
    probe.capture()
    assert all(p.grad is None for p in model.parameters())


@pytest.fixture
def live_server() -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(
        create_app(), host="127.0.0.1", port=port, log_level="warning"
    )
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uv.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    uv.should_exit = True
    thread.join(10)


def test_end_to_end_with_the_real_backend(live_server: str) -> None:
    torch.manual_seed(0)
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = torch.randn(16, 3), torch.randn(16, 4)
    probe = nv.ActivationProbe(model, {"relu": model.act}, x[:1])
    with nv.start_run("e2e", total_epochs=3, url=live_server) as run:
        assert run.enabled
        for epoch in range(1, 4):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            run.log(epoch=epoch, metrics={"loss": loss}, activations=probe.capture())
    with urllib.request.urlopen(f"{live_server}/api/v1/runs/{run.run_id}") as resp:
        info = json.loads(resp.read())
    assert info["status"] == "completed"
    assert info["steps_logged"] == 3 and info["epoch"] == 3
    assert run.dropped == 0


# Health statistics and two-way control (ADR 0008) ----------------------------


class ControlServer(FakeServer):
    """A FakeServer that hands out queued commands and records acks."""

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[dict[str, Any]] = []
        self.acks: dict[int, dict[str, Any]] = {}

    def issue(self, kind: str, value: float | None = None) -> int:
        command_id = len(self.pending) + len(self.acks) + 1
        self.pending.append(
            {
                "command_id": command_id,
                "kind": kind,
                "value": value,
                "status": "pending",
            }
        )
        return command_id

    def __call__(
        self, method: str, url: str, body: Any, headers: dict[str, str]
    ) -> tuple[int, Any]:
        if url.endswith("/commands") and method == "GET":
            self.requests.append((method, url, body, headers))
            return 200, {"commands": list(self.pending)}
        if url.endswith("/ack"):
            self.requests.append((method, url, body, headers))
            command_id = int(url.split("/")[-2])
            self.acks[command_id] = body
            self.pending = [c for c in self.pending if c["command_id"] != command_id]
            return 200, {}
        return super().__call__(method, url, body, headers)

    def wait_for_ack(self, command_id: int, timeout: float = 5.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while command_id not in self.acks:
            assert time.monotonic() < deadline, f"no ack for command {command_id}"
            time.sleep(0.02)
        return self.acks[command_id]


@pytest.fixture
def control_server() -> ControlServer:
    return ControlServer()


def controlled_run(server: ControlServer, **kwargs: Any) -> nv.Run:
    return nv.start_run(
        "ctl",
        total_epochs=2,
        transport=server,
        control=True,
        flush_interval=0.02,
        control_interval=0.02,
        **kwargs,
    )


def poll_until(run: nv.Run, condition: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        run.poll()
        time.sleep(0.01)


def test_controls_are_declared_only_when_enabled(control_server: ControlServer) -> None:
    optimizer = torch.optim.SGD(Net().parameters(), lr=0.1)
    nv.start_run("plain", total_epochs=1, transport=control_server, optimizer=optimizer)
    assert "controls" not in control_server.requests[0][2]
    controlled_run(control_server, optimizer=optimizer, on_checkpoint=lambda: "a.pt")
    assert control_server.requests[-1][2]["controls"] == [
        "pause",
        "resume",
        "stop",
        "set_lr",
        "checkpoint",
    ]


def test_set_lr_and_checkpoint_apply_in_the_training_thread(
    control_server: ControlServer,
) -> None:
    optimizer = torch.optim.Adam(Net().parameters(), lr=0.1)
    saved: list[str] = []

    def save() -> str:
        saved.append(threading.current_thread().name)
        return "epoch-1.pt"

    run = controlled_run(control_server, optimizer=optimizer, on_checkpoint=save)
    lr_id = control_server.issue("set_lr", 0.005)
    ck_id = control_server.issue("checkpoint")
    poll_until(run, lambda: run.commands_applied == 2)
    assert optimizer.param_groups[0]["lr"] == 0.005
    assert saved == [threading.current_thread().name]  # Not the client's thread.
    assert control_server.wait_for_ack(lr_id) == {
        "status": "applied",
        "paused": False,
        "learning_rate": 0.005,
        "message": "Learning rate set to 0.005. A scheduler may override it.",
    }
    assert (
        control_server.wait_for_ack(ck_id)["message"] == "Checkpoint saved: epoch-1.pt."
    )
    run.finish()


def test_pause_blocks_poll_until_resume(control_server: ControlServer) -> None:
    run = controlled_run(control_server)
    run.log(epoch=1, metrics={"loss": 1.0})
    pause_id = control_server.issue("pause")

    def resume_soon() -> None:
        # The pause is acknowledged while the training thread is blocked.
        control_server.wait_for_ack(pause_id)
        time.sleep(0.3)
        control_server.issue("resume")

    threading.Thread(target=resume_soon, daemon=True).start()
    deadline = time.monotonic() + 5
    while True:
        started = time.monotonic()
        assert run.poll() is True
        blocked = time.monotonic() - started
        if pause_id in control_server.acks:
            break
        assert time.monotonic() < deadline, "the pause never arrived"
        time.sleep(0.01)
    assert blocked >= 0.25  # The poll that paused waited for the resume.
    assert not run.paused
    assert control_server.acks[pause_id]["paused"] is True
    run.finish()


def test_stop_makes_poll_false_and_finishes_cancelled(
    control_server: ControlServer,
) -> None:
    with controlled_run(control_server) as run:
        control_server.issue("stop")
        poll_until(run, lambda: run.stop_requested)
        assert run.poll() is False and run.should_stop
    assert control_server.finish_body()["status"] == "cancelled"


def test_unsupported_or_invalid_commands_are_rejected(
    control_server: ControlServer,
) -> None:
    run = controlled_run(control_server)  # No optimizer, no checkpoint function.
    resume_id = control_server.issue("resume")
    lr_id = control_server.issue("set_lr", 0.1)
    poll_until(run, lambda: len(control_server.acks) == 2)
    assert control_server.acks[resume_id]["status"] == "rejected"
    assert control_server.acks[resume_id]["message"] == "The run is not paused."
    assert control_server.acks[lr_id]["status"] == "rejected"
    run.finish()


def test_failing_checkpoint_reports_only_the_error_type(
    control_server: ControlServer,
) -> None:
    def broken() -> None:
        raise OSError("disk full at /home/me/secret")

    run = controlled_run(control_server, on_checkpoint=broken)
    cid = control_server.issue("checkpoint")
    poll_until(run, lambda: cid in control_server.acks)
    assert control_server.acks[cid]["message"] == "on_checkpoint raised OSError."
    run.finish()


def test_poll_is_free_without_controls(server: FakeServer) -> None:
    run = nv.start_run("plain", total_epochs=1, transport=server)
    assert run.poll() is True
    run.finish()
    assert not any(url.endswith("/commands") for _, url, _, _ in server.requests)


def test_non_finite_metrics_are_reported_by_name(server: FakeServer) -> None:
    run = nv.start_run("nan", total_epochs=1, transport=server)
    run.log(epoch=1, metrics={"loss": float("nan"), "accuracy": 0.5})
    run.finish()
    (event,) = server.events()
    assert event["metrics"] == {"accuracy": 0.5}
    assert event["health"]["nonfinite_metrics"] == ["loss"]


class Deep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(self.fc2(self.relu(self.fc1(x))))


def test_health_monitor_measures_gradients_and_updates() -> None:
    torch.manual_seed(0)
    model = Deep()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monitor = nv.HealthMonitor(model, optimizer=optimizer)
    x = torch.randn(16, 4)

    first = monitor.collect()  # Before any step: weights only.
    assert "grad_norm" not in first and first["parameters"]["fc1"]["parameters"] == 40
    before = model.fc1.weight.detach().clone(), model.fc1.bias.detach().clone()

    optimizer.zero_grad()
    model(x).pow(2).mean().backward()
    expected_grad = math.sqrt(
        float(model.fc1.weight.grad.pow(2).sum() + model.fc1.bias.grad.pow(2).sum())
    )
    optimizer.step()
    health = monitor.collect()

    fc1 = health["parameters"]["fc1"]
    assert fc1["grad_norm"] == pytest.approx(expected_grad, rel=1e-5)
    weight, bias = model.fc1.weight.detach(), model.fc1.bias.detach()
    weight_norm = math.sqrt(float(weight.pow(2).sum() + bias.pow(2).sum()))
    update = math.sqrt(
        float((weight - before[0]).pow(2).sum() + (bias - before[1]).pow(2).sum())
    )
    assert fc1["weight_norm"] == pytest.approx(weight_norm, rel=1e-5)
    assert fc1["update_ratio"] == pytest.approx(update / weight_norm, rel=1e-4)
    total = math.sqrt(sum(g["grad_norm"] ** 2 for g in health["parameters"].values()))
    assert health["grad_norm"] == pytest.approx(total, rel=1e-6)
    assert health["learning_rate"] == 0.1 and health["nonfinite_grads"] == 0


def test_health_monitor_finds_dead_and_saturated_units() -> None:
    model = Deep()
    with torch.no_grad():
        # Half of fc1's units can never be positive: dead ReLUs.
        model.fc1.weight[:4] = 0
        model.fc1.bias[:4] = -1
        # Huge outputs saturate tanh.
        model.fc2.weight.mul_(1000)
    monitor = nv.HealthMonitor(model, probe_batch=torch.randn(32, 4))
    health = monitor.collect()
    assert health["activations"]["relu"]["dead_fraction"] == pytest.approx(
        0.5, abs=0.13
    )
    assert health["activations"]["relu"]["dead_fraction"] >= 0.5
    assert health["activations"]["tanh"]["saturated_fraction"] > 0.5
    assert model.training and not model.relu._forward_hooks


def test_health_monitor_counts_non_finite_gradients() -> None:
    model = Deep()
    model(torch.randn(2, 4)).sum().backward()
    with torch.no_grad():
        model.fc2.weight.grad[0, 0] = float("nan")
    health = nv.HealthMonitor(model).collect()
    assert health["nonfinite_grads"] == 1
    assert "grad_norm" not in health["parameters"]["fc2"]


def test_health_monitor_merges_many_groups() -> None:
    model = nn.Sequential(
        *[nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)) for _ in range(20)]
    )
    monitor = nv.HealthMonitor(model)
    groups = monitor.collect()["parameters"]
    assert len(groups) == 20  # 40 Linear layers merged into their 20 blocks.
    assert sum(g["parameters"] for g in groups.values()) == 40 * 6


def test_health_end_to_end_with_controls(live_server: str) -> None:
    torch.manual_seed(0)
    model = Deep()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    monitor = nv.HealthMonitor(
        model, optimizer=optimizer, probe_batch=torch.randn(8, 4)
    )
    x, y = torch.randn(32, 4), torch.randn(32, 3)
    run = nv.start_run(
        "health-e2e",
        total_epochs=50,
        url=live_server,
        control=True,
        optimizer=optimizer,
        control_interval=0.05,
        flush_interval=0.05,
    )

    def post(path: str, body: Any) -> Any:
        request = urllib.request.Request(
            f"{live_server}/api/v1/runs/{run.run_id}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    post("/commands", {"kind": "set_lr", "value": 0.01})
    epoch = 0
    deadline = time.monotonic() + 10
    while run.poll():
        epoch += 1
        optimizer.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
        run.log(epoch=epoch, metrics={"loss": 0.5}, health=monitor.collect())
        if epoch == 3:
            post("/commands", {"kind": "stop"})
        assert time.monotonic() < deadline, "the stop command never arrived"
        time.sleep(0.02)
    run.finish()
    with urllib.request.urlopen(f"{live_server}/api/v1/runs/{run.run_id}") as resp:
        info = json.loads(resp.read())
    assert info["status"] == "cancelled"
    assert info["learning_rate"] == 0.01 and optimizer.param_groups[0]["lr"] == 0.01
    assert info["pending_commands"] == 0
    # A constant loss for 20 logged steps is a plateau; nothing else is wrong.
    assert epoch < 20 or info["alerts"] >= 1


# System statistics -----------------------------------------------------------


class FakeSystem:
    def __init__(self, stats: dict[str, Any]) -> None:
        self.stats = stats

    def sample(self) -> dict[str, Any]:
        return dict(self.stats)


def test_sends_timing_eta_and_throughput(server: FakeServer) -> None:
    run = nv.start_run("t", total_epochs=10, url="http://nv.test", transport=server)
    clock = iter([100.0, 101.0, 102.0, 104.0])
    run._started = 100.0
    run._system = FakeSystem({})
    import neurovision_client

    real = neurovision_client.time.monotonic
    neurovision_client.time.monotonic = lambda: next(clock)  # type: ignore[assignment]
    try:
        for epoch in (1, 2, 3, 4):
            run.log(epoch=epoch, metrics={"loss": 1.0}, samples=64)
    finally:
        neurovision_client.time.monotonic = real  # type: ignore[assignment]
    run.finish()
    systems = [e["system"] for e in server.events()]
    assert systems[0] == {"elapsed_s": 0.0}
    assert systems[1]["step_time_s"] == 1.0 and systems[1]["samples_per_s"] == 64.0
    # Epochs 1..4 over 4 s: 0.75 epochs/s, 6 epochs left -> 8 s.
    assert systems[3]["eta_s"] == pytest.approx(8.0)
    assert systems[3]["elapsed_s"] == 4.0


def test_cleans_system_stats(server: FakeServer) -> None:
    run = start(server)
    run._system = FakeSystem(
        {
            "cpu_percent": 250.0,
            "ram_used_bytes": float("nan"),
            "device": "cuda:0",
            "gpus": [
                {"index": 0, "name": "GPU", "util_percent": 87.0, "power_w": 55.5},
                {"index": 1, "util_percent": float("inf"), "temperature_c": 61.0},
            ],
        }
    )
    run.log(epoch=1, metrics={"loss": 1.0})
    run.finish()
    system = server.events()[0]["system"]
    assert "ram_used_bytes" not in system
    assert "cpu_percent" not in system  # Out of range, so dropped.
    assert system["device"] == "cuda:0"
    assert system["gpus"][0] == {
        "index": 0,
        "name": "GPU",
        "util_percent": 87.0,
        "power_w": 55.5,
    }
    assert system["gpus"][1] == {"index": 1, "temperature_c": 61.0}


def test_system_monitor_reads_this_machine() -> None:
    stats = nv.SystemMonitor().sample()
    # psutil may be missing; when present the numbers are sane.
    if "cpu_percent" in stats:
        assert 0 <= stats["cpu_percent"] <= 100
        assert stats["ram_total_bytes"] > 0


def test_track_system_off_sends_nothing(server: FakeServer) -> None:
    run = start(server, track_system=False)
    run.log(epoch=1, metrics={"loss": 1.0})
    run.finish()
    assert "system" not in server.events()[0]


def test_eta_follows_fractional_progress(server: FakeServer) -> None:
    run = nv.start_run("t", total_epochs=2, url="http://nv.test", transport=server)
    clock = iter([10.0, 11.0])
    run._system = FakeSystem({})
    import neurovision_client

    real = neurovision_client.time.monotonic
    neurovision_client.time.monotonic = lambda: next(clock)  # type: ignore[assignment]
    try:
        # Both inside epoch 1: a quarter of an epoch per second.
        run.log(epoch=1, metrics={"loss": 1.0}, progress=0.25)
        run.log(epoch=1, metrics={"loss": 1.0}, progress=0.5)
    finally:
        neurovision_client.time.monotonic = real  # type: ignore[assignment]
    run.finish()
    assert server.events()[1]["system"]["eta_s"] == pytest.approx(6.0)
