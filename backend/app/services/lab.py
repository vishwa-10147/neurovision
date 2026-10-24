"""Launch and track Neural Lab training scripts.

Safety: only the scripts listed in TEMPLATES, from this repository's
sdk/examples folder, are ever run; every parameter is checked against its
template's type and bounds and passed as a separate argument (no shell).
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from app.schemas.lab import LabJob, ParamSpec, Template

EXAMPLES = Path(__file__).resolve().parents[3] / "sdk" / "examples"
MAX_OUTPUT_LINES = 40
MAX_JOBS_KEPT = 20


def _int(
    name: str, label: str, default: int, low: int, high: int, help: str | None = None
) -> ParamSpec:
    return ParamSpec(
        name=name,
        label=label,
        kind="int",
        default=default,
        min=low,
        max=high,
        help=help,
    )


def _float(
    name: str,
    label: str,
    default: float,
    low: float,
    high: float,
    help: str | None = None,
) -> ParamSpec:
    return ParamSpec(
        name=name,
        label=label,
        kind="float",
        default=default,
        min=low,
        max=high,
        help=help,
    )


def _common(epochs: int, lr: float, label: str = "Epochs") -> list[ParamSpec]:
    return [
        _int("epochs", label, epochs, 1, 1000),
        _float("lr", "Learning rate", lr, 0.00001, 1.0),
        _float(
            "delay",
            "Slow motion (seconds per epoch)",
            0.0,
            0.0,
            5.0,
            "pause after each epoch, to watch",
        ),
    ]


@dataclass(frozen=True)
class Script:
    file: str
    template: Template


def _template(
    id: str,
    file: str,
    label: str,
    family: str,
    description: str,
    shows: list[str],
    params: list[ParamSpec],
) -> Script:
    return Script(
        file,
        Template(
            id=id,
            label=label,
            family=family,
            description=description,
            shows=shows,
            params=params,
        ),
    )


TEMPLATES: dict[str, Script] = {
    s.template.id: s
    for s in [
        _template(
            "mlp",
            "train_spiral.py",
            "MLP on a spiral",
            "Dense network",
            "A 2 → 32 → 16 → 3 network (ReLU, tanh) classifying a generated 3-arm spiral.",
            ["graph colored by layer activity", "two hidden layers' values"],
            [
                _int("epochs", "Epochs", 150, 1, 1000),
                _float("lr", "Learning rate", 0.003, 0.00001, 1.0),
                _float("delay", "Slow motion (seconds per epoch)", 0.05, 0.0, 5.0),
            ],
        ),
        _template(
            "cnn",
            "train_digits_cnn.py",
            "CNN on handwritten digits",
            "Convolutional",
            "Convolution layers (channels doubling each layer), max-pooling and a classifier on scikit-learn's "
            "1,797 real 8×8 digit images.",
            ["graph colored by layer activity", "ReLU outputs of each convolution"],
            [
                _int("blocks", "Convolution layers", 2, 1, 4),
                _int("channels", "Channels (first layer)", 16, 2, 64),
                _int("hidden", "Classifier units", 64, 8, 512),
                *_common(30, 0.002),
            ],
        ),
        _template(
            "rnn",
            "train_lstm.py",
            "Recurrent network forecasting a series",
            "Recurrent",
            "An LSTM, GRU or plain RNN reads 30 past values of a generated seasonal series and predicts the next.",
            ["forecast vs actual on held-out data", "hidden state at every time step"],
            [
                ParamSpec(
                    name="cell",
                    label="Cell",
                    kind="choice",
                    default="lstm",
                    choices=["lstm", "gru", "rnn"],
                ),
                _int("hidden", "Hidden units", 32, 2, 256),
                _int("layers", "Stacked layers", 1, 1, 4),
                *_common(30, 0.003),
            ],
        ),
        _template(
            "transformer",
            "train_attention.py",
            "Transformer (self-attention)",
            "Attention",
            "Transformer blocks (multi-head self-attention + feed-forward) learn whether a sequence's first "
            "token appears again: a task that needs attention.",
            [
                "Q, K, V, softmax and feed-forward in the graph",
                "attention block activations",
            ],
            [
                _int(
                    "dim", "Model width", 32, 8, 256, "must be divisible by the heads"
                ),
                _int("heads", "Heads", 4, 1, 16),
                _int("blocks", "Blocks", 1, 1, 6),
                *_common(40, 0.002),
            ],
        ),
        _template(
            "autoencoder",
            "train_autoencoder.py",
            "Autoencoder on digits",
            "Autoencoder",
            "Compresses each 64-pixel digit to a small code and rebuilds it; the loss is the reconstruction error.",
            ["the code values of one digit"],
            [_int("code", "Code size", 8, 1, 32), *_common(60, 0.003)],
        ),
        _template(
            "vae",
            "train_vae.py",
            "Variational autoencoder (2D latent space)",
            "Generative",
            "Maps digits to a 2D latent space and back; the clusters of digits form without labels.",
            [
                "latent space of held-out digits by class",
                "originals vs reconstructions",
            ],
            _common(80, 0.002),
        ),
        _template(
            "gan",
            "train_gan.py",
            "GAN drawing digits",
            "Generative",
            "A generator and a discriminator compete; the generator learns to draw 8×8 digits from noise.",
            ["16 generated digits from fixed noise, every epoch"],
            _common(150, 0.0005),
        ),
        _template(
            "gnn",
            "train_gnn.py",
            "Graph neural network (karate club)",
            "Graph",
            "A graph convolutional network predicts each member's faction in Zachary's karate club from "
            "friendships and only 2 labels.",
            ["2D embedding of every member", "normalized adjacency matrix"],
            _common(200, 0.01),
        ),
        _template(
            "diffusion",
            "train_diffusion.py",
            "Diffusion model drawing a spiral",
            "Generative",
            "Learns to remove noise; sampling turns random noise into spiral-shaped points in 100 steps.",
            ["freshly generated points every few epochs"],
            _common(300, 0.001),
        ),
        _template(
            "cartpole",
            "train_cartpole.py",
            "Reinforcement learning: CartPole",
            "Reinforcement",
            "A policy network learns, by REINFORCE, to balance a pole on a cart (classic physics, 500 steps max).",
            ["pole angle over the latest episode"],
            _common(150, 0.01, label="Iterations (10 episodes each)"),
        ),
    ]
}


class LabError(ValueError):
    """A launch request does not fit its template, or the lab is busy or off."""


def validate(
    template_id: str, params: dict[str, float | str]
) -> tuple[Script, list[str]]:
    """The script and its command-line arguments, or LabError."""
    script = TEMPLATES.get(template_id)
    if script is None:
        raise LabError(f"Unknown template '{template_id}'.")
    specs = {p.name: p for p in script.template.params}
    unknown = sorted(set(params) - set(specs))
    if unknown:
        raise LabError(f"Unknown parameters for {template_id}: {', '.join(unknown)}.")
    args: list[str] = []
    for spec in script.template.params:
        value = params.get(spec.name, spec.default)
        if spec.kind == "choice":
            if value not in (spec.choices or []):
                raise LabError(
                    f"{spec.label} must be one of {', '.join(spec.choices or [])}."
                )
            text = str(value)
        else:
            if isinstance(value, str) or isinstance(value, bool):
                raise LabError(f"{spec.label} must be a number.")
            number = float(value)
            if spec.kind == "int" and number != int(number):
                raise LabError(f"{spec.label} must be a whole number.")
            if not (spec.min <= number <= spec.max):  # type: ignore[operator]
                raise LabError(
                    f"{spec.label} must be between {spec.min:g} and {spec.max:g}."
                )
            text = str(int(number)) if spec.kind == "int" else repr(number)
        args += [f"--{spec.name}", text]
    if template_id == "transformer":
        dim, heads = int(params.get("dim", 32)), int(params.get("heads", 4))
        if dim % heads:
            raise LabError(
                f"Model width ({dim}) must be divisible by the number of heads ({heads})."
            )
    return script, args


@dataclass
class _Job:
    job_id: str
    template: str
    run_name: str
    process: subprocess.Popen[str]
    started_at: float
    output: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_OUTPUT_LINES))
    stopped: bool = False

    def view(self) -> LabJob:
        code = self.process.poll()
        status = (
            "running"
            if code is None
            else "stopped"
            if self.stopped
            else "finished"
            if code == 0
            else "failed"
        )
        return LabJob(
            job_id=self.job_id,
            template=self.template,
            run_name=self.run_name,
            status=status,
            exit_code=code,
            started_at=self.started_at,
            output=list(self.output),
        )


class LabManager:
    def __init__(self, python: str | None, max_jobs: int, enabled: bool) -> None:
        self.python = python or sys.executable
        self.max_jobs = max_jobs
        self.enabled = enabled
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def launch(
        self, template_id: str, params: dict[str, float | str], base_url: str
    ) -> LabJob:
        if not self.enabled:
            raise LabError(
                "The Neural Lab is turned off on this server (NEUROVISION_LAB_ENABLED)."
            )
        script, args = validate(template_id, params)
        path = EXAMPLES / script.file
        if not path.is_file():
            raise LabError(
                f"{script.file} is missing from {EXAMPLES}; the lab needs the repository's sdk folder."
            )
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.process.poll() is None)
            if running >= self.max_jobs:
                raise LabError(
                    f"{running} lab runs are already training; stop one or wait."
                )
            job_id = f"lab_{secrets.token_hex(4)}"
            run_name = f"{script.template.label} · {job_id[-4:]}"
            command = [
                self.python,
                "-u",
                str(path),
                "--url",
                base_url,
                "--name",
                run_name,
                *args,
            ]
            process = subprocess.Popen(
                command,
                cwd=str(EXAMPLES),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                # The child's output is read as UTF-8 whatever the console uses.
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            job = _Job(job_id, template_id, run_name, process, time.time())
            self._jobs[job_id] = job
            self._forget_old()
        threading.Thread(target=self._read, args=(job,), daemon=True).start()
        return job.view()

    def _read(self, job: _Job) -> None:
        assert job.process.stdout is not None
        for line in job.process.stdout:
            job.output.append(line.rstrip()[:300])

    def _forget_old(self) -> None:
        finished = [j for j in self._jobs.values() if j.process.poll() is not None]
        for job in sorted(finished, key=lambda j: j.started_at)[
            : max(0, len(self._jobs) - MAX_JOBS_KEPT)
        ]:
            del self._jobs[job.job_id]

    def jobs(self) -> list[LabJob]:
        with self._lock:
            return [
                j.view()
                for j in sorted(self._jobs.values(), key=lambda j: -j.started_at)
            ]

    def stop(self, job_id: str) -> LabJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.process.poll() is None:
            job.stopped = True
            job.process.terminate()
            try:
                job.process.wait(5)
            except subprocess.TimeoutExpired:
                job.process.kill()
        return job.view()

    def shutdown(self) -> None:
        for job_id in list(self._jobs):
            self.stop(job_id)
