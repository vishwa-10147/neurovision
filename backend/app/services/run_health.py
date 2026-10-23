"""Training-health rules for external runs (ADR 0008).

Pure and deterministic: feed every logged step to `HealthRules.observe` and
it returns the alerts that start at that step. An alert fires once when its
condition starts and re-arms when the condition clears, so a persistent
problem produces one alert rather than one per step.

Thresholds are conventional heuristics; alerts are signals to check, and
their messages always quote the measured numbers.
"""

import statistics
from collections import deque
from dataclasses import dataclass

from app.schemas.events import AlertKind, StepHealth, StepSystem

# Gradient norm far above its recent median.
EXPLODING_FACTOR = 10.0
GRAD_WINDOW = 20
MIN_HISTORY = 5
# A group's gradient norm this small is effectively zero.
VANISHING_NORM = 1e-7
# Update ratio ‖Δw‖/‖w‖ bounds; around 1e-3 is commonly considered healthy.
UPDATE_RATIO_HIGH = 0.1
UPDATE_RATIO_LOW = 1e-6
DEAD_FRACTION = 0.5
SATURATED_FRACTION = 0.5
# Conditions on layers must hold for this many health steps in a row.
PERSIST_STEPS = 3
LOSS_WINDOW = 10
SPIKE_FACTOR = 3.0
# Best loss must improve by more than this fraction within PLATEAU_STEPS.
PLATEAU_IMPROVEMENT = 0.01
PLATEAU_STEPS = 20
OVERFIT_STEPS = 5
# A GPU this busy or less, for this many logged steps, is mostly waiting.
GPU_UNDERUSED_PERCENT = 20.0
GPU_UNDERUSED_STEPS = 5
GPU_MEMORY_HIGH = 0.95

LOSS_NAMES = ("loss", "train_loss", "training_loss")
VAL_LOSS_NAMES = ("val_loss", "valid_loss", "validation_loss")

Severity = str


@dataclass(frozen=True)
class Finding:
    kind: AlertKind
    severity: Severity
    message: str
    advice: str
    layer: str | None = None


def _first(metrics: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _g(value: float) -> str:
    return f"{value:.3g}"


class HealthRules:
    def __init__(self) -> None:
        self._active: set[tuple[str, str | None]] = set()
        self._streaks: dict[tuple[str, str | None], int] = {}
        self._grad_norms: deque[float] = deque(maxlen=GRAD_WINDOW)
        self._losses: deque[float] = deque(maxlen=LOSS_WINDOW)
        self._best_loss: float | None = None
        self._since_improvement = 0
        self._pairs: deque[tuple[float, float]] = deque(maxlen=OVERFIT_STEPS + 1)

    def observe(
        self,
        metrics: dict[str, float],
        health: StepHealth | None,
        system: StepSystem | None = None,
    ) -> list[Finding]:
        found: list[Finding] = []
        if health is not None:
            self._gradients(health, found)
            self._parameters(health, found)
            self._activations(health, found)
        self._loss(metrics, health, found)
        if system is not None:
            self._system(system, found)
        return found

    # Rules ---------------------------------------------------------------

    def _gradients(self, health: StepHealth, found: list[Finding]) -> None:
        self._edge(
            found,
            ("non_finite_gradients", None),
            health.nonfinite_grads > 0,
            Finding(
                "non_finite_gradients",
                "critical",
                f"{health.nonfinite_grads:,} gradient values are NaN or infinite.",
                "Lower the learning rate, clip gradients "
                "(torch.nn.utils.clip_grad_norm_), or check the inputs and loss "
                "for invalid values (log of 0, division by 0). With mixed "
                "precision, GradScaler skips overflowed steps on purpose: "
                "collect health only on steps it applied.",
            ),
        )
        norm = health.grad_norm
        if norm is None:
            return
        exploding = False
        median = 0.0
        if len(self._grad_norms) >= MIN_HISTORY:
            median = statistics.median(self._grad_norms)
            exploding = median > 0 and norm > EXPLODING_FACTOR * median
        self._edge(
            found,
            ("exploding_gradients", None),
            exploding,
            Finding(
                "exploding_gradients",
                "warning",
                f"The gradient norm jumped to {_g(norm)}, "
                f"{norm / median if median else 0:.0f}× its recent median of {_g(median)}.",
                "Clip gradients (torch.nn.utils.clip_grad_norm_) or lower the "
                "learning rate; check for a bad batch.",
            ),
        )
        self._grad_norms.append(norm)

    def _parameters(self, health: StepHealth, found: list[Finding]) -> None:
        for name, group in health.parameters.items():
            if group.grad_norm is not None:
                self._persistent(
                    found,
                    ("vanishing_gradients", name),
                    group.grad_norm < VANISHING_NORM,
                    Finding(
                        "vanishing_gradients",
                        "warning",
                        f"Gradients in {name} have been about zero "
                        f"(norm {_g(group.grad_norm)}) for {PERSIST_STEPS} measurements.",
                        "The layer is not learning: check for dead or saturated "
                        "activations before it, frozen parameters, or a detached "
                        "graph; residual connections or normalization can help.",
                        name,
                    ),
                )
            ratio = group.update_ratio
            if ratio is None:
                continue
            self._edge(
                found,
                ("update_ratio_high", name),
                ratio > UPDATE_RATIO_HIGH,
                Finding(
                    "update_ratio_high",
                    "warning",
                    f"Updates in {name} are {ratio:.1e} of the weights' size; above "
                    "about 1e-2 training is usually unstable.",
                    "Lower the learning rate (a ratio near 1e-3 is a common target).",
                    name,
                ),
            )
            self._persistent(
                found,
                ("update_ratio_low", name),
                ratio < UPDATE_RATIO_LOW,
                Finding(
                    "update_ratio_low",
                    "info",
                    f"Updates in {name} are only {ratio:.1e} of the weights' size.",
                    "The learning rate may be too low for this layer, or the layer "
                    "is effectively frozen.",
                    name,
                ),
            )

    def _activations(self, health: StepHealth, found: list[Finding]) -> None:
        for name, layer in health.activations.items():
            if layer.dead_fraction is not None:
                self._persistent(
                    found,
                    ("dead_units", name),
                    layer.dead_fraction >= DEAD_FRACTION,
                    Finding(
                        "dead_units",
                        "warning",
                        f"{layer.dead_fraction:.0%} of units in {name} output zero for "
                        "every probe example (dead ReLUs).",
                        "Lower the learning rate, check the initialization, or use "
                        "LeakyReLU or GELU.",
                        name,
                    ),
                )
            if layer.saturated_fraction is not None:
                self._persistent(
                    found,
                    ("saturated_units", name),
                    layer.saturated_fraction >= SATURATED_FRACTION,
                    Finding(
                        "saturated_units",
                        "warning",
                        f"{layer.saturated_fraction:.0%} of {layer.kind} outputs in "
                        f"{name} are saturated, so their gradients are near zero.",
                        "Normalize the inputs, scale the initialization down, add "
                        "batch normalization, or lower the learning rate.",
                        name,
                    ),
                )

    def _loss(
        self,
        metrics: dict[str, float],
        health: StepHealth | None,
        found: list[Finding],
    ) -> None:
        nonfinite = list(health.nonfinite_metrics) if health is not None else []
        self._edge(
            found,
            ("non_finite_loss", None),
            bool(nonfinite),
            Finding(
                "non_finite_loss",
                "critical",
                f"{', '.join(nonfinite[:4])} became NaN or infinite.",
                "Stop or pause the run: lower the learning rate, clip gradients, "
                "and check the data and loss for invalid values.",
            ),
        )
        loss = _first(metrics, LOSS_NAMES)
        if loss is None:
            return
        spike = False
        median = 0.0
        if len(self._losses) >= MIN_HISTORY:
            median = statistics.median(self._losses)
            spike = median > 0 and loss > SPIKE_FACTOR * median
        self._edge(
            found,
            ("loss_spike", None),
            spike,
            Finding(
                "loss_spike",
                "warning",
                f"The loss spiked to {_g(loss)}, "
                f"{loss / median if median else 0:.1f}× its recent median of {_g(median)}.",
                "A spike often comes from a too-high learning rate or a bad "
                "batch; clip gradients or lower the learning rate if it repeats.",
            ),
        )
        self._losses.append(loss)

        if self._best_loss is None or loss < self._best_loss * (
            1 - PLATEAU_IMPROVEMENT
        ):
            self._best_loss = loss
            self._since_improvement = 0
        else:
            self._since_improvement += 1
        best = self._best_loss
        self._edge(
            found,
            ("loss_plateau", None),
            self._since_improvement >= PLATEAU_STEPS,
            Finding(
                "loss_plateau",
                "info",
                f"The loss has not improved by more than {PLATEAU_IMPROVEMENT:.0%} "
                f"for {PLATEAU_STEPS} logged steps (best {_g(best)}).",
                "Try lowering the learning rate, or stop if the validation metric "
                "is flat too.",
            ),
        )

        val = _first(metrics, VAL_LOSS_NAMES)
        if val is None:
            return
        self._pairs.append((loss, val))
        overfitting = False
        if len(self._pairs) == self._pairs.maxlen:
            pairs = list(self._pairs)
            val_rising = all(
                b[1] > a[1] for a, b in zip(pairs, pairs[1:], strict=False)
            )
            train_falling = pairs[-1][0] < pairs[0][0]
            overfitting = val_rising and train_falling
        self._edge(
            found,
            ("overfitting", None),
            overfitting,
            Finding(
                "overfitting",
                "warning",
                f"Validation loss rose for {OVERFIT_STEPS} logged steps in a row "
                f"(now {_g(val)}) while training loss kept falling.",
                "The model is starting to overfit: save a checkpoint now, and "
                "consider early stopping, more regularization, or more data.",
            ),
        )

    def _system(self, system: StepSystem, found: list[Finding]) -> None:
        # An idle GPU matters only if the script trains on it (CPU-only
        # scripts on a machine with a GPU report no CUDA device).
        on_gpu = (system.device or "").startswith("cuda")
        for gpu in system.gpus:
            label = f"GPU {gpu.index}" + (f" ({gpu.name})" if gpu.name else "")
            if gpu.util_percent is not None and on_gpu:
                self._persistent(
                    found,
                    ("gpu_underused", str(gpu.index)),
                    gpu.util_percent <= GPU_UNDERUSED_PERCENT,
                    Finding(
                        "gpu_underused",
                        "info",
                        f"{label} has been only {gpu.util_percent:.0f}% busy for "
                        f"{GPU_UNDERUSED_STEPS} logged steps.",
                        "The GPU is mostly waiting: the data loader or CPU "
                        "preprocessing is likely the bottleneck (try more DataLoader "
                        "workers, pin_memory=True, or a larger batch), or the model "
                        "is running on the CPU.",
                        label,
                    ),
                    steps=GPU_UNDERUSED_STEPS,
                )
            if gpu.mem_used_bytes is not None and gpu.mem_total_bytes:
                share = gpu.mem_used_bytes / gpu.mem_total_bytes
                self._edge(
                    found,
                    ("gpu_memory_high", str(gpu.index)),
                    share >= GPU_MEMORY_HIGH,
                    Finding(
                        "gpu_memory_high",
                        "warning",
                        f"{label} memory is {share:.0%} full "
                        f"({gpu.mem_used_bytes / 2**30:.1f} of "
                        f"{gpu.mem_total_bytes / 2**30:.1f} GiB).",
                        "An out-of-memory error is close: lower the batch size, use "
                        "mixed precision (torch.autocast), or gradient checkpointing.",
                        label,
                    ),
                )

    # Edge detection ------------------------------------------------------

    def _edge(
        self,
        found: list[Finding],
        key: tuple[str, str | None],
        condition: bool,
        finding: Finding,
    ) -> None:
        """Report `finding` when `condition` becomes true; re-arm when false."""
        if condition and key not in self._active:
            self._active.add(key)
            found.append(finding)
        elif not condition:
            self._active.discard(key)

    def _persistent(
        self,
        found: list[Finding],
        key: tuple[str, str | None],
        condition: bool,
        finding: Finding,
        steps: int = PERSIST_STEPS,
    ) -> None:
        """Like `_edge`, but only after `steps` consecutive steps."""
        streak = self._streaks.get(key, 0) + 1 if condition else 0
        self._streaks[key] = streak
        self._edge(found, key, streak >= steps, finding)
