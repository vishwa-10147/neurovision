"""Framework callback integration tests against a real NeuroVision backend."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import neurovision_callbacks as callbacks
import pytest
import torch
import uvicorn
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from app.main import create_app


@pytest.fixture
def live_server() -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(
        create_app(), host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(10)


def run_info(url: str, run_id: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{url}/api/v1/runs/{run_id}") as response:
        return json.loads(response.read())


def post(url: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def replay(url: str, run_id: str) -> list[dict[str, Any]]:
    from websockets.sync.client import connect

    events: list[dict[str, Any]] = []
    with connect(
        url.replace("http://", "ws://") + f"/api/v1/training/{run_id}/events",
        open_timeout=5,
        close_timeout=5,
    ) as websocket:
        while True:
            event = json.loads(websocket.recv())
            events.append(event)
            if event["event_type"] in {
                "training.completed",
                "training.cancelled",
                "training.failed",
            }:
                return events


def steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == "training.step"]


def test_normalize_cleans_framework_metric_names() -> None:
    result = callbacks._normalize(
        {
            "loss": torch.tensor(0.8),
            "loss_epoch": 0.7,
            "loss_step": 0.9,
            "train_loss": 0.6,
            "train/loss": 0.5,
            "val/loss": 0.4,
            "eval_accuracy": 0.75,
            "eval_runtime": 2.0,
            "total_flos": 100.0,
            "vector": [1.0, 2.0],
        }
    )
    assert result["loss"] == pytest.approx(0.8)
    assert result["val_loss"] == pytest.approx(0.4)
    assert result["val_accuracy"] == pytest.approx(0.75)
    assert set(result) == {"loss", "val_loss", "val_accuracy"}


def test_session_merges_same_step_and_flushes_everything() -> None:
    requests: list[tuple[str, str, Any, dict[str, str]]] = []

    def transport(
        method: str, url: str, body: Any, headers: dict[str, str]
    ) -> tuple[int, Any]:
        requests.append((method, url, body, headers))
        if url.endswith("/api/v1/runs"):
            return 201, {"run": {"run_id": "run_callback"}, "write_token": "token"}
        return (202, {"commands": []}) if url.endswith("/events") else (200, {})

    session = callbacks._Session(
        name="merge",
        url="http://test",
        total_epochs=2,
        control=False,
        checkpoint_dir=None,
        health=False,
        activation_layers=None,
        probe_batch=None,
        run_options={"transport": transport, "track_system": False},
    )
    session.start(total_epochs=2)
    session.record(step=1, epoch=1, metrics={"train_loss": 1.0})
    session.record(step=1, epoch=1, metrics={"eval_loss": 0.5})
    session.record(step=2, epoch=2, metrics={"loss": 0.25})
    session.finish()
    event_batches = [body for _, url, body, _ in requests if url.endswith("/events")]
    events = [event for batch in event_batches for event in batch["events"]]
    assert [event["step"] for event in events] == [1, 2]
    assert events[0]["metrics"] == {"loss": 1.0, "val_loss": 0.5}
    assert events[1]["metrics"] == {"loss": 0.25}


def test_importing_callbacks_does_not_import_frameworks() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "sdk")
    code = (
        "import sys; import neurovision_callbacks; "
        "print(any(name in sys.modules for name in ('lightning', 'transformers', 'keras')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize(
    ("callback", "blocked", "hint"),
    [
        (
            "LightningCallback",
            ["lightning", "pytorch_lightning"],
            "neurovision-client[lightning]",
        ),
        (
            "HuggingFaceCallback",
            ["transformers"],
            "neurovision-client[transformers]",
        ),
        ("KerasCallback", ["keras"], "neurovision-client[keras]"),
    ],
)
def test_missing_framework_gives_an_install_hint(
    callback: str, blocked: list[str], hint: str
) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "sdk")
    code = (
        "import sys\n"
        f"for name in {blocked!r}: sys.modules[name] = None\n"
        "try:\n"
        f"    from neurovision_callbacks import {callback}\n"
        "except ModuleNotFoundError as error:\n"
        "    print(error)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert hint in result.stdout
    assert result.stdout.startswith(f"{callback} needs")


def _lightning_model(crash_at: int | None = None) -> Any:
    import lightning.pytorch as pl

    class Model(pl.LightningModule):
        example_input_array = torch.zeros(1, 2)

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
            self.crash_at = crash_at

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
            if self.crash_at is not None and batch_idx == self.crash_at:
                raise ValueError("intentional test failure")
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("train_loss", loss, on_step=False, on_epoch=True)
            return loss

        def validation_step(self, batch: Any, batch_idx: int) -> None:
            del batch_idx
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("val_loss", loss, on_step=False, on_epoch=True)

        def configure_optimizers(self) -> Any:
            return torch.optim.SGD(self.parameters(), lr=0.05)

    return Model()


def _lightning_trainer(callback: Any, *, max_epochs: int = 3, **kwargs: Any) -> Any:
    import lightning.pytorch as pl

    return pl.Trainer(
        max_epochs=max_epochs,
        limit_train_batches=4,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[callback],
        **kwargs,
    )


def _lightning_data() -> tuple[DataLoader[Any], DataLoader[Any]]:
    torch.manual_seed(0)
    x = torch.randn(32, 2)
    y = (x[:, 0] + x[:, 1] > 0).long()
    return DataLoader(TensorDataset(x[:24], y[:24]), batch_size=6), DataLoader(
        TensorDataset(x[24:], y[24:]), batch_size=8
    )


def test_lightning_callback_end_to_end(live_server: str) -> None:
    callback = callbacks.LightningCallback(
        url=live_server,
        run_options={"track_system": False, "flush_interval": 0.01},
        probe_batch=torch.zeros(1, 2),
        activation_layers=["net.1"],
    )
    trainer = _lightning_trainer(callback)
    trainer.fit(_lightning_model(), *_lightning_data())
    assert callback.run is not None
    info = run_info(live_server, callback.run.run_id)
    events = replay(live_server, callback.run.run_id)
    logged = steps(events)
    assert info["status"] == "completed"
    assert len(logged) == 3 and all(
        b["step"] > a["step"] for a, b in zip(logged, logged[1:], strict=False)
    )
    assert all("loss" in event["metrics"] for event in logged)
    assert all("val_loss" in event["metrics"] for event in logged)
    assert any("health" in event for event in logged)


def test_lightning_sends_each_epochs_own_metrics(
    live_server: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression: validation ran before on_train_epoch_end and was flushed
    there, so each step carried the previous epoch's train_loss (on_epoch=True
    metrics only exist after on_train_epoch_end) and the last one was lost."""
    import lightning.pytorch as pl

    truth: list[dict[str, float]] = []

    class Truth(pl.Callback):
        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            del pl_module
            truth.append({k: float(v) for k, v in trainer.callback_metrics.items()})

    callback = callbacks.LightningCallback(
        url=live_server,
        health=False,
        run_options={"track_system": False, "flush_interval": 0.01},
    )
    trainer = _lightning_trainer(callback, max_epochs=4)
    trainer.callbacks.append(Truth())
    with caplog.at_level("WARNING"):
        trainer.fit(_lightning_model(), *_lightning_data())
    assert callback.run is not None
    logged = steps(replay(live_server, callback.run.run_id))
    assert [event["epoch"] for event in logged] == [1, 2, 3, 4]
    for event, expected in zip(logged, truth, strict=True):
        assert event["metrics"]["loss"] == pytest.approx(expected["train_loss"])
        assert event["metrics"]["val_loss"] == pytest.approx(expected["val_loss"])
    assert "did not increase" not in caplog.text
    assert "ignored step" not in caplog.text


def test_lightning_callback_stop_finishes_cancelled(live_server: str) -> None:
    callback = callbacks.LightningCallback(
        url=live_server,
        control=True,
        run_options={
            "track_system": False,
            "flush_interval": 0.01,
            "control_interval": 0.01,
        },
    )

    def stop_soon() -> None:
        deadline = time.monotonic() + 5
        while callback.run is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if callback.run is not None:
            time.sleep(0.1)
            post(
                live_server,
                f"/api/v1/runs/{callback.run.run_id}/commands",
                {"kind": "stop"},
            )

    stopper = threading.Thread(target=stop_soon, daemon=True)
    stopper.start()
    trainer = _lightning_trainer(callback, max_epochs=30)
    trainer.fit(_lightning_model(), *_lightning_data())
    stopper.join(5)
    assert callback.run is not None
    info = run_info(live_server, callback.run.run_id)
    assert info["status"] == "cancelled"
    assert info["epoch"] < 30


def test_lightning_callback_crash_is_reported(live_server: str) -> None:
    callback = callbacks.LightningCallback(
        url=live_server,
        run_options={"track_system": False, "flush_interval": 0.01},
    )
    trainer = _lightning_trainer(callback, max_epochs=3)
    with pytest.raises(ValueError, match="intentional test failure"):
        trainer.fit(_lightning_model(crash_at=1), *_lightning_data())
    assert callback.run is not None
    info = run_info(live_server, callback.run.run_id)
    events = replay(live_server, callback.run.run_id)
    assert info["status"] == "failed"
    assert events[-1]["message"] == "The training script raised ValueError."


class TinyRows(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        torch.manual_seed(1)
        self.input_ids = torch.randint(0, 50, (32, 8))
        self.attention_mask = torch.ones(32, 8, dtype=torch.long)
        self.labels = torch.randint(0, 2, (32,))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def test_huggingface_callback_end_to_end_without_downloads(
    live_server: str, tmp_path: Path
) -> None:
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=50,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=2,
        )
    )
    callback = callbacks.HuggingFaceCallback(
        control=True,
        url=live_server,
        run_options={"track_system": False, "flush_interval": 0.01},
    )
    args = TrainingArguments(
        output_dir=str(tmp_path / "hf-output"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    Trainer(
        model=model,
        args=args,
        train_dataset=TinyRows(),
        eval_dataset=TinyRows(),
        callbacks=[callback],
    ).train()
    assert callback.run is not None
    info = run_info(live_server, callback.run.run_id)
    events = replay(live_server, callback.run.run_id)
    logged = steps(events)
    assert info["status"] == "completed"
    assert logged and any("val_loss" in event["metrics"] for event in logged)
    assert all("eval_runtime" not in event["metrics"] for event in logged)
    assert info["controls"] == ["pause", "resume", "stop"]


def _keras_model() -> Any:
    import keras

    model = keras.Sequential(
        [
            keras.Input((3,)),
            keras.layers.Dense(8, activation="relu", name="hidden"),
            keras.layers.Dense(2, name="output"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        run_eagerly=True,
    )
    return model


def _keras_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    x = torch.randn(64, 3)
    y = (x[:, 0] > 0).long()
    return x, y


def test_keras_callback_end_to_end_and_set_lr(live_server: str) -> None:
    model = _keras_model()
    callback = callbacks.KerasCallback(
        control=True,
        url=live_server,
        run_options={
            "track_system": False,
            "flush_interval": 0.01,
            "control_interval": 0.01,
        },
    )

    def set_lr_soon() -> None:
        deadline = time.monotonic() + 5
        while callback.run is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if callback.run is not None:
            time.sleep(0.05)
            post(
                live_server,
                f"/api/v1/runs/{callback.run.run_id}/commands",
                {"kind": "set_lr", "value": 0.005},
            )

    thread = threading.Thread(target=set_lr_soon, daemon=True)
    thread.start()
    x, y = _keras_data()
    model.fit(
        x,
        y,
        epochs=3,
        batch_size=4,
        validation_split=0.25,
        verbose=0,
        callbacks=[callback],
    )
    thread.join(5)
    assert callback.run is not None
    info = run_info(live_server, callback.run.run_id)
    logged = steps(replay(live_server, callback.run.run_id))
    assert info["status"] == "completed"
    assert logged and any("val_loss" in event["metrics"] for event in logged)
    assert "set_lr" in info["controls"]
    import keras

    assert float(
        keras.ops.convert_to_numpy(model.optimizer.learning_rate)
    ) == pytest.approx(0.005)


def test_keras_schedule_does_not_offer_set_lr(live_server: str) -> None:
    import keras

    model = _keras_model()
    model.optimizer = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.ExponentialDecay(
            0.01, decay_steps=10, decay_rate=0.9
        )
    )
    callback = callbacks.KerasCallback(
        control=True,
        url=live_server,
        run_options={"track_system": False, "flush_interval": 0.01},
    )
    x, y = _keras_data()
    model.fit(x, y, epochs=1, batch_size=16, verbose=0, callbacks=[callback])
    assert callback.run is not None
    assert "set_lr" not in run_info(live_server, callback.run.run_id)["controls"]
