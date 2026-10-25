"""Optional callbacks for streaming framework training to NeuroVision.

The framework imports in this module are deliberately lazy.  Importing
``neurovision_callbacks`` therefore works in a plain Python environment, and
each callback only requires the framework it wraps when that callback class is
first requested::

    from neurovision_callbacks import LightningCallback
    trainer = pl.Trainer(callbacks=[LightningCallback()])

The callbacks share the same small session implementation so metrics have the
same names, command behavior and bounded payloads as the raw SDK client.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import neurovision_client as nv

logger = logging.getLogger("neurovision_callbacks")

__all__ = ["HuggingFaceCallback", "KerasCallback", "LightningCallback"]  # noqa: F822

_DROPPED_METRICS = {
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
    "total_flos",
    "train_runtime",
    "epoch",
}


def _metric_value(value: Any) -> float | None:
    """Return a finite scalar metric, or ``None`` for non-scalar values."""
    if isinstance(value, bytes | Mapping | list | tuple):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    try:
        if hasattr(value, "numel") and int(value.numel()) != 1:
            return None
        if hasattr(value, "size") and not callable(value.size):
            size = value.size
            if size != 1:
                return None
        if hasattr(value, "item"):
            value = value.item()
        if not isinstance(value, numbers.Real):
            return None
        result = float(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    return result if math.isfinite(result) else None


def _metric_name(name: str) -> str:
    """Map framework naming conventions to the names used by Live Runs."""
    if name.endswith("_epoch"):
        name = name[:-6]
    elif name.endswith("_step"):
        name = name[:-5]
    if name.startswith("eval_"):
        return "val_" + name[5:]
    if name.startswith("eval/"):
        return "val_" + name[5:]
    if name == "val/loss":
        return "val_loss"
    if name.startswith("val/"):
        return "val_" + name[4:]
    if name in {"train_loss", "train/loss"}:
        return "loss"
    if name == "loss_epoch":
        return "loss"
    if name == "loss_step":
        return "loss"
    return name.replace("/", "_")


def _is_validation(name: str) -> bool:
    """Whether a raw metric name is a validation metric (val_x, val/x, eval_x)."""
    return _metric_name(str(name)).startswith("val_")


def _normalize(metrics: Mapping[str, Any], framework: str = "") -> dict[str, float]:
    """Clean framework logs into compact, chart-friendly scalar metrics.

    ``framework`` is intentionally accepted even though the current rules are
    shared.  It keeps the helper's public testing surface explicit and leaves
    room for framework-specific cleanup without making the callbacks diverge.
    """
    del framework
    values: dict[str, tuple[int, float]] = {}
    has_direct_loss = "loss" in metrics and _metric_value(metrics["loss"]) is not None
    has_epoch_loss = (
        "loss_epoch" in metrics and _metric_value(metrics["loss_epoch"]) is not None
    )
    priorities = {"loss": 0, "loss_epoch": 1, "train_loss": 2, "train/loss": 2}
    for raw_name, raw_value in metrics.items():
        name = str(raw_name)
        if name in _DROPPED_METRICS:
            continue
        value = _metric_value(raw_value)
        if value is None:
            continue
        if name in {"loss_step", "train_loss_step", "train/loss_step"} and (
            has_epoch_loss or "train_loss_epoch" in metrics
        ):
            continue
        clean_name = _metric_name(name)
        if clean_name == "loss" and has_direct_loss and name != "loss":
            continue
        priority = priorities.get(name, 3)
        previous = values.get(clean_name)
        if previous is None or priority < previous[0]:
            values[clean_name] = (priority, value)
    return {name: value for name, (_, value) in values.items()}


class _Session:
    """Framework-independent run lifecycle and step-merging logic."""

    def __init__(
        self,
        *,
        name: str | None,
        url: str | None,
        total_epochs: int | None,
        control: bool,
        checkpoint_dir: str | Path | None,
        health: bool,
        activation_layers: Iterable[str] | None,
        probe_batch: Any,
        run_options: Mapping[str, Any] | None,
    ) -> None:
        self.name = name
        self.url = url
        self.total_epochs = total_epochs
        self.control = control
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.health_enabled = health
        self.activation_layers = list(activation_layers or [])
        self.probe_batch = probe_batch
        self.run_options = dict(run_options or {})
        self.run: nv.Run | None = None
        self.health_monitor: nv.HealthMonitor | None = None
        self.activation_probe: nv.ActivationProbe | None = None
        self.activation_capture: Callable[[], Mapping[str, Any]] | None = None
        self.pending: dict[str, Any] | None = None
        self._latest_health: Mapping[str, Any] | None = None
        self._latest_activations: Mapping[str, Any] | None = None
        self._finished = False
        self._warned: set[str] = set()

    def start(
        self,
        *,
        total_epochs: int,
        model: Any = None,
        optimizer: Any = None,
        control_optimizer: Any = None,
        example_input: Any = None,
        on_checkpoint: Callable[[], Any] | None = None,
        allow_set_lr: bool = False,
        link_graph: bool = False,
    ) -> None:
        if self.run is not None:
            return
        options = dict(self.run_options)
        options.update(
            {
                "url": self.url,
                "total_epochs": max(1, int(total_epochs)),
                "control": self.control,
                "optimizer": control_optimizer if allow_set_lr else None,
                "on_checkpoint": on_checkpoint,
            }
        )
        if link_graph and model is not None and example_input is not None:
            options.update(model=model, example_input=example_input)
        self.total_epochs = max(1, int(total_epochs))
        self.run = nv.start_run(self.name or type(model).__name__, **options)
        if self.health_enabled and model is not None and optimizer is not None:
            try:
                self.health_monitor = nv.HealthMonitor(
                    model, optimizer=optimizer, probe_batch=self.probe_batch
                )
            except Exception as error:
                self._warn_once("health", f"health monitoring is disabled ({error}).")
        if (
            self.activation_layers
            and self.probe_batch is not None
            and model is not None
        ):
            try:
                self.activation_probe = nv.ActivationProbe.by_name(
                    model, self.activation_layers, self.probe_batch
                )
            except Exception as error:
                self._warn_once(
                    "activations", f"activation probing is disabled ({error})."
                )

    def set_activation_capture(
        self, capture: Callable[[], Mapping[str, Any]] | None
    ) -> None:
        self.activation_capture = capture

    def capture_observations(self) -> None:
        if self.run is None or not self.run.enabled:
            return
        if self.health_monitor is not None:
            try:
                self._latest_health = self.health_monitor.collect()
            except Exception as error:
                self._warn_once(
                    "health-capture", f"could not collect health ({error})."
                )
        capture = self.activation_capture
        if capture is None and self.activation_probe is not None:
            capture = self.activation_probe.capture
        if capture is not None:
            try:
                self._latest_activations = capture()
            except Exception as error:
                self._warn_once(
                    "activation-capture", f"could not capture activations ({error})."
                )

    def record(
        self,
        *,
        step: int,
        epoch: int,
        metrics: Mapping[str, Any] | None,
        progress: float | None = None,
        samples: int | None = None,
    ) -> None:
        clean = _normalize(metrics or {})
        step = max(0, int(step))
        entry = {
            "step": step,
            "epoch": max(0, int(epoch)),
            "progress": progress,
            "samples": samples,
            "metrics": clean,
        }
        if self.pending is None:
            self.pending = entry
        elif step == self.pending["step"]:
            self.pending["metrics"].update(clean)
            if progress is not None:
                self.pending["progress"] = progress
            if samples is not None:
                self.pending["samples"] = samples
            self.pending["epoch"] = max(self.pending["epoch"], entry["epoch"])
        elif step > self.pending["step"]:
            self.flush()
            self.pending = entry
        else:
            self._warn_once(
                "ordering",
                f"ignored step {step} after pending step {self.pending['step']}.",
            )

    def flush(self) -> None:
        if self.pending is None or self.run is None:
            return
        entry = self.pending
        self.pending = None
        self.run.log(
            epoch=entry["epoch"],
            step=entry["step"],
            metrics=entry["metrics"],
            activations=self._latest_activations,
            health=self._latest_health,
            progress=entry["progress"],
            samples=entry["samples"],
        )

    def poll(self) -> bool:
        return self.run is None or self.run.poll()

    def finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        self.flush()
        if self.run is None:
            return
        if error is None:
            self.run.finish()
        elif isinstance(error, KeyboardInterrupt):
            self.run.finish("cancelled", "The training script was interrupted.")
        else:
            self.run.finish(
                "failed", f"The training script raised {type(error).__name__}."
            )

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning("NeuroVision callback '%s': %s", self.name or "run", message)


def _checkpoint_path(
    directory: Path, step: int, suffix: str, *, as_directory: bool = False
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"neurovision-step{max(0, step)}{suffix}"
    if as_directory:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _make_lightning() -> type[Any]:
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl  # type: ignore[no-redef]

    class LightningCallback(pl.Callback):  # type: ignore[misc, valid-type]
        """Stream a PyTorch Lightning training run to NeuroVision."""

        def __init__(
            self,
            name: str | None = None,
            *,
            url: str | None = None,
            total_epochs: int | None = None,
            control: bool = False,
            checkpoint_dir: str | Path | None = None,
            health: bool = True,
            activation_layers: Iterable[str] | None = None,
            probe_batch: Any = None,
            log_every_n_steps: int | None = None,
            run_options: Mapping[str, Any] | None = None,
            link_graph: bool = False,
        ) -> None:
            self._session = _Session(
                name=name,
                url=url,
                total_epochs=total_epochs,
                control=control,
                checkpoint_dir=checkpoint_dir,
                health=health,
                activation_layers=activation_layers,
                probe_batch=probe_batch,
                run_options=run_options,
            )
            self.log_every_n_steps = log_every_n_steps
            self.total_epochs_override = total_epochs
            self.link_graph = link_graph
            self._trainer: Any = None
            self._model: Any = None

        @property
        def run(self) -> nv.Run | None:
            return self._session.run

        def _start(self, trainer: Any, model: Any) -> None:
            if getattr(trainer, "is_global_zero", True) is False:
                return
            max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)
            if self.total_epochs_override is not None:
                epochs = self.total_epochs_override
            elif max_epochs > 0:
                epochs = max_epochs
            else:
                max_steps = int(getattr(trainer, "max_steps", 0) or 0)
                batches = getattr(trainer, "num_training_batches", 0)
                if (
                    max_steps > 0
                    and isinstance(batches, numbers.Real)
                    and float(batches) > 0
                ):
                    epochs = math.ceil(max_steps / float(batches))
                else:
                    epochs = 1
                    logger.warning(
                        "NeuroVision Lightning callback could not determine total epochs; using 1."
                    )
            optimizers = list(getattr(trainer, "optimizers", []) or [])
            optimizer = optimizers[0] if optimizers else None
            allow_set_lr = not bool(getattr(trainer, "lr_scheduler_configs", []))
            example = getattr(model, "example_input_array", None)
            self._trainer, self._model = trainer, model
            self._session.start(
                total_epochs=epochs,
                model=model,
                optimizer=optimizer,
                control_optimizer=optimizer,
                example_input=example,
                on_checkpoint=self._save_checkpoint
                if self._session.checkpoint_dir is not None
                else None,
                allow_set_lr=allow_set_lr,
                link_graph=self.link_graph and example is not None,
            )

        def _is_logging_step(self, step: int) -> bool:
            return step > 0 and (
                self.log_every_n_steps is None
                or step % max(1, int(self.log_every_n_steps)) == 0
            )

        @staticmethod
        def _metrics(trainer: Any) -> dict[str, Any]:
            metrics: dict[str, Any] = {}
            for source in (
                getattr(trainer, "logged_metrics", {}),
                getattr(trainer, "callback_metrics", {}),
            ):
                if isinstance(source, Mapping):
                    metrics.update(source)
            return metrics

        def on_train_start(self, trainer: Any, pl_module: Any) -> None:
            if not getattr(trainer, "sanity_checking", False):
                self._start(trainer, pl_module)

        def on_train_batch_start(
            self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int
        ) -> None:
            del pl_module, batch, batch_idx
            if (
                not getattr(trainer, "sanity_checking", False)
                and not self._session.poll()
            ):
                trainer.should_stop = True

        def on_before_zero_grad(
            self, trainer: Any, pl_module: Any, optimizer: Any
        ) -> None:
            del pl_module, optimizer
            if getattr(trainer, "sanity_checking", False):
                return
            next_step = int(getattr(trainer, "global_step", 0)) + 1
            if self._is_logging_step(next_step):
                self._session.capture_observations()

        def on_train_batch_end(
            self,
            trainer: Any,
            pl_module: Any,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            del pl_module, batch
            if getattr(trainer, "sanity_checking", False):
                return
            step = int(getattr(trainer, "global_step", 0))
            batches = getattr(trainer, "num_training_batches", 0)
            last_batch = (
                isinstance(batches, numbers.Real)
                and float(batches) > 0
                and batch_idx + 1 >= int(float(batches))
            )
            if (self.log_every_n_steps is None and last_batch) or (
                self.log_every_n_steps is not None and self._is_logging_step(step)
            ):
                fraction = (
                    (batch_idx + 1) / float(batches)
                    if isinstance(batches, numbers.Real) and float(batches) > 0
                    else 0.0
                )
                metrics = self._metrics(trainer)
                if "loss" not in metrics:
                    if isinstance(outputs, Mapping) and "loss" in outputs:
                        metrics["loss"] = outputs["loss"]
                    elif _metric_value(outputs) is not None:
                        metrics["loss"] = outputs
                self._session.record(
                    step=step,
                    epoch=int(getattr(trainer, "current_epoch", 0)) + 1,
                    progress=float(getattr(trainer, "current_epoch", 0)) + fraction,
                    metrics=metrics,
                )

        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            # Lightning runs the epoch's validation before this hook, and
            # computes epoch-level training metrics (self.log(on_epoch=True))
            # only here, so this is the one point where both are current.
            del pl_module
            if getattr(trainer, "sanity_checking", False):
                return
            self._session.record(
                step=int(getattr(trainer, "global_step", 0)),
                epoch=int(getattr(trainer, "current_epoch", 0)) + 1,
                progress=float(getattr(trainer, "current_epoch", 0)) + 1.0,
                metrics=self._metrics(trainer),
            )
            self._session.flush()

        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            # Merge only: flushing here would send last epoch's training
            # metrics, and on_train_epoch_end's same-step update would be lost.
            del pl_module
            if getattr(trainer, "sanity_checking", False):
                return
            metrics = self._metrics(trainer)
            self._session.record(
                step=int(getattr(trainer, "global_step", 0)),
                epoch=int(getattr(trainer, "current_epoch", 0)) + 1,
                progress=float(getattr(trainer, "current_epoch", 0)) + 1.0,
                metrics={k: v for k, v in metrics.items() if _is_validation(k)},
            )

        def on_exception(
            self, trainer: Any, pl_module: Any, exception: BaseException
        ) -> None:
            del trainer, pl_module
            self._session.finish(exception)

        def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
            del trainer, pl_module
            self._session.finish()

        def _save_checkpoint(self) -> str:
            if self._trainer is None or self._session.checkpoint_dir is None:
                raise RuntimeError("No Lightning trainer or checkpoint directory.")
            step = int(getattr(self._trainer, "global_step", 0))
            path = _checkpoint_path(self._session.checkpoint_dir, step, ".ckpt")
            self._trainer.save_checkpoint(str(path))
            return path.name

    LightningCallback.__name__ = "LightningCallback"
    return LightningCallback


def _make_huggingface() -> type[Any]:
    from transformers import TrainerCallback

    class HuggingFaceCallback(TrainerCallback):  # type: ignore[misc]
        """Stream a Hugging Face ``Trainer`` run to NeuroVision."""

        def __init__(
            self,
            name: str | None = None,
            *,
            url: str | None = None,
            total_epochs: int | None = None,
            control: bool = False,
            checkpoint_dir: str | Path | None = None,
            health: bool = True,
            activation_layers: Iterable[str] | None = None,
            probe_batch: Any = None,
            log_every_n_steps: int | None = None,
            run_options: Mapping[str, Any] | None = None,
        ) -> None:
            self._session = _Session(
                name=name,
                url=url,
                total_epochs=total_epochs,
                control=control,
                checkpoint_dir=checkpoint_dir,
                health=health,
                activation_layers=activation_layers,
                probe_batch=probe_batch,
                run_options=run_options,
            )
            self.log_every_n_steps = log_every_n_steps
            self.total_epochs_override = total_epochs
            self._model: Any = None
            self._state: Any = None
            self._args: Any = None
            self._last_log_step = 0

        @property
        def run(self) -> nv.Run | None:
            return self._session.run

        def on_train_begin(
            self,
            args: Any,
            state: Any,
            control: Any,
            model: Any = None,
            optimizer: Any = None,
            **kwargs: Any,
        ) -> None:
            del kwargs
            if getattr(state, "is_world_process_zero", True) is False:
                return
            epochs_value = (
                self.total_epochs_override
                or getattr(state, "num_train_epochs", None)
                or getattr(args, "num_train_epochs", 1)
            )
            epochs = max(1, math.ceil(float(epochs_value)))
            interval = self.log_every_n_steps
            if interval is None:
                configured = getattr(args, "logging_steps", 0)
                interval = (
                    int(float(configured))
                    if isinstance(configured, numbers.Real)
                    else 0
                )
            self.log_every_n_steps = interval if interval > 0 else None
            self._model, self._state, self._args = model, state, args
            health_optimizer = getattr(optimizer, "optimizer", optimizer)
            self._session.start(
                total_epochs=epochs,
                model=model,
                optimizer=health_optimizer,
                control_optimizer=None,
                on_checkpoint=self._save_checkpoint
                if self._session.checkpoint_dir is not None
                else None,
                allow_set_lr=False,
            )

        def _is_logging_step(self, step: int) -> bool:
            return (
                self.log_every_n_steps is not None
                and step > 0
                and step % int(self.log_every_n_steps) == 0
            )

        def on_step_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            del args, state, kwargs
            if not self._session.poll():
                control.should_training_stop = True
            return control

        def on_optimizer_step(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            del args, kwargs
            next_step = int(getattr(state, "global_step", 0)) + 1
            if self._is_logging_step(next_step):
                self._session.capture_observations()
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Mapping[str, Any],
            **kwargs: Any,
        ) -> Any:
            del control, kwargs
            if "train_runtime" in logs:
                return None
            step = int(getattr(state, "global_step", 0))
            epoch_value = getattr(state, "epoch", None)
            epoch = max(1, math.ceil(float(epoch_value or 1)))
            progress = float(epoch_value) if epoch_value is not None else float(epoch)
            batch_size = int(getattr(args, "train_batch_size", 0) or 0)
            samples = None
            if step > self._last_log_step and batch_size:
                accumulation = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
                samples = (step - self._last_log_step) * batch_size * accumulation
            self._session.record(
                step=step,
                epoch=epoch,
                progress=progress,
                samples=samples,
                metrics=logs,
            )
            self._last_log_step = max(self._last_log_step, step)
            return None

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            del args, control, kwargs
            pending = self._session.pending
            if pending is not None and pending["step"] < int(
                getattr(state, "global_step", 0)
            ):
                self._session.flush()
            return None

        def on_epoch_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            del args, control, kwargs
            # Evaluation logging can follow this hook at the same global step.
            # Leave the pending entry open so eval_loss merges into it.
            return None

        def on_train_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            del args, state, control, kwargs
            self._session.finish()
            return None

        def _save_checkpoint(self) -> str:
            if self._model is None or self._session.checkpoint_dir is None:
                raise RuntimeError("No Hugging Face model or checkpoint directory.")
            step = int(getattr(self._state, "global_step", 0))
            path = _checkpoint_path(
                self._session.checkpoint_dir, step, "", as_directory=True
            )
            self._model.save_pretrained(str(path))
            return path.name

    HuggingFaceCallback.__name__ = "HuggingFaceCallback"
    return HuggingFaceCallback


def _make_keras() -> type[Any]:
    import keras

    class _KerasLr:
        def __init__(self, optimizer: Any) -> None:
            self.optimizer = optimizer

        def __getitem__(self, key: str) -> float:
            if key != "lr":
                raise KeyError(key)
            return float(keras.ops.convert_to_numpy(self.optimizer.learning_rate))

        def __setitem__(self, key: str, value: Any) -> None:
            if key != "lr":
                raise KeyError(key)
            self.optimizer.learning_rate = value

    class _KerasOptimizer:
        def __init__(self, optimizer: Any) -> None:
            self.param_groups = [_KerasLr(optimizer)]

    class KerasCallback(keras.callbacks.Callback):  # type: ignore[misc]
        """Stream a Keras ``Model.fit`` run to NeuroVision."""

        def __init__(
            self,
            name: str | None = None,
            *,
            url: str | None = None,
            total_epochs: int | None = None,
            control: bool = False,
            checkpoint_dir: str | Path | None = None,
            health: bool = False,
            activation_layers: Iterable[str] | None = None,
            probe_batch: Any = None,
            log_every_n_steps: int | None = None,
            run_options: Mapping[str, Any] | None = None,
        ) -> None:
            del health  # Keras has no PyTorch-style gradients for HealthMonitor.
            self._session = _Session(
                name=name,
                url=url,
                total_epochs=total_epochs,
                control=control,
                checkpoint_dir=checkpoint_dir,
                health=False,
                activation_layers=activation_layers,
                probe_batch=probe_batch,
                run_options=run_options,
            )
            self.log_every_n_steps = log_every_n_steps
            self.total_epochs_override = total_epochs
            self._steps_per_epoch = 0
            self._epoch = 0
            self._step = 0
            self._probe_model: Any = None

        @property
        def run(self) -> nv.Run | None:
            return self._session.run

        def on_train_begin(self, logs: Mapping[str, Any] | None = None) -> None:
            del logs
            model = self.model
            optimizer = getattr(model, "optimizer", None)
            learning_rate = getattr(optimizer, "_learning_rate", None)
            schedule_type = keras.optimizers.schedules.LearningRateSchedule
            allow_set_lr = not isinstance(learning_rate, schedule_type)
            epochs = self.total_epochs_override or int(
                self.params.get("epochs", 1) or 1
            )
            self._steps_per_epoch = int(self.params.get("steps", 0) or 0)
            run_optimizer = _KerasOptimizer(optimizer) if allow_set_lr else None
            self._session.start(
                total_epochs=epochs,
                model=model,
                optimizer=None,
                control_optimizer=run_optimizer,
                on_checkpoint=self._save_checkpoint
                if self._session.checkpoint_dir is not None
                else None,
                allow_set_lr=allow_set_lr,
            )
            self._build_probe()

        def _build_probe(self) -> None:
            names = self._session.activation_layers
            if not names or self._session.probe_batch is None:
                return
            try:
                inputs = self.model.inputs
                if not inputs:
                    raise ValueError("the model has no functional inputs")
                layers = [self.model.get_layer(name) for name in names]
                probe_model = keras.Model(inputs, [layer.output for layer in layers])

                def capture() -> Mapping[str, list[float]]:
                    outputs = probe_model(self._session.probe_batch, training=False)
                    if not isinstance(outputs, list | tuple):
                        outputs = [outputs]
                    captured: dict[str, list[float]] = {}
                    for name, output in zip(names, outputs, strict=False):
                        array = keras.ops.convert_to_numpy(output)
                        if getattr(array, "ndim", 0) > 0:
                            array = array[0]
                        values = array.reshape(-1).tolist()[: nv.MAX_VALUES_PER_LAYER]
                        captured[name] = [float(value) for value in values]
                    return captured

                self._probe_model = probe_model
                self._session.set_activation_capture(capture)
            except Exception as error:
                logger.warning(
                    "NeuroVision Keras callback: activation probing is disabled (%s).",
                    error,
                )

        def on_train_batch_begin(
            self, batch: int, logs: Mapping[str, Any] | None = None
        ) -> None:
            del batch, logs
            if not self._session.poll():
                self.model.stop_training = True

        def on_train_batch_end(
            self, batch: int, logs: Mapping[str, Any] | None = None
        ) -> None:
            if self.log_every_n_steps is None or self._steps_per_epoch <= 0:
                return
            step = self._epoch * self._steps_per_epoch + int(batch) + 1
            if step % max(1, int(self.log_every_n_steps)) != 0:
                return
            self._step = step
            self._session.capture_observations()
            self._session.record(
                step=step,
                epoch=self._epoch + 1,
                progress=self._epoch + (int(batch) + 1) / self._steps_per_epoch,
                metrics=logs,
            )

        def on_epoch_begin(
            self, epoch: int, logs: Mapping[str, Any] | None = None
        ) -> None:
            del logs
            self._epoch = int(epoch)

        def on_epoch_end(
            self, epoch: int, logs: Mapping[str, Any] | None = None
        ) -> None:
            step = (
                (int(epoch) + 1) * self._steps_per_epoch
                if self._steps_per_epoch
                else int(epoch) + 1
            )
            self._step = step
            self._session.capture_observations()
            self._session.record(
                step=step,
                epoch=int(epoch) + 1,
                progress=float(int(epoch) + 1),
                metrics=logs or {},
            )
            self._session.flush()

        def on_train_end(self, logs: Mapping[str, Any] | None = None) -> None:
            del logs
            self._session.finish()

        def _save_checkpoint(self) -> str:
            if self._session.checkpoint_dir is None:
                raise RuntimeError("No Keras checkpoint directory.")
            path = _checkpoint_path(self._session.checkpoint_dir, self._step, ".keras")
            self.model.save(str(path))
            return path.name

    KerasCallback.__name__ = "KerasCallback"
    return KerasCallback


# Missing top-level package -> how to fix it, per callback.
_INSTALL_HINTS: dict[str, dict[str, str]] = {
    "LightningCallback": {
        "lightning": 'PyTorch Lightning: pip install "neurovision-client[lightning]"',
        "pytorch_lightning": 'PyTorch Lightning: pip install "neurovision-client[lightning]"',
    },
    "HuggingFaceCallback": {
        "transformers": 'Hugging Face Transformers: pip install "neurovision-client[transformers]"',
    },
    "KerasCallback": {
        "keras": 'Keras 3: pip install "neurovision-client[keras]"',
        # Keras defaults to the TensorFlow backend.
        "tensorflow": "a Keras backend. Install TensorFlow, or use PyTorch by setting "
        "KERAS_BACKEND=torch before importing keras",
    },
}


def __getattr__(name: str) -> Any:
    factories: dict[str, Callable[[], type[Any]]] = {
        "LightningCallback": _make_lightning,
        "HuggingFaceCallback": _make_huggingface,
        "KerasCallback": _make_keras,
    }
    factory = factories.get(name)
    if factory is None:
        raise AttributeError(name)
    try:
        callback = factory()
    except ModuleNotFoundError as error:
        hint = _INSTALL_HINTS.get(name, {}).get((error.name or "").split(".")[0])
        if hint is None:
            raise
        raise ModuleNotFoundError(f"{name} needs {hint}", name=error.name) from error
    globals()[name] = callback
    return callback
