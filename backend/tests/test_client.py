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
    run = start(server)
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
