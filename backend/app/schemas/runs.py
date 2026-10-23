"""External runs: training scripts outside NeuroVision that stream their own
metrics and activations through POST /api/v1/runs/{run_id}/events.

Logged steps use the same bounds as training.step events, so the existing
event stream and UI can follow them unchanged.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from app.schemas.events import (
    MAX_ACTIVATION_LAYERS,
    MAX_VALUES_PER_LAYER,
    CommandKind,
    Message,
    Metrics,
    NonNegativeFloat,
    RunCommand,
    StepHealth,
)
from app.schemas.graph import Identifier

MAX_EVENTS_PER_REQUEST = 32
# Steps a run may log in total; the replay history stays bounded too.
MAX_STEPS_PER_RUN = 10_000
MAX_TOTAL_EPOCHS = 1_000_000
# Commands waiting for the script to pick them up (ADR 0008).
MAX_PENDING_COMMANDS = 16
MAX_LEARNING_RATE = 10.0

# "cancelled": the script stopped because NeuroVision asked it to.
RunStatus = Literal["running", "completed", "failed", "cancelled"]
NonNegativeInt = Annotated[int, Field(ge=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunCreate(_Strict):
    name: Annotated[str, Field(min_length=1, max_length=80)]
    total_epochs: Annotated[int, Field(ge=1, le=MAX_TOTAL_EPOCHS)]
    # Optional link to a model already in NeuroVision, e.g. an imported ONNX
    # export of the network being trained.
    model_id: (
        Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
        | None
    ) = None
    # Commands the script will apply; none means the run cannot be controlled.
    controls: Annotated[list[CommandKind], Field(max_length=5)] = Field(
        default_factory=list
    )


class LoggedStep(_Strict):
    """One training.step without the fields the server stamps itself."""

    epoch: NonNegativeInt
    # Strictly increasing within a run, e.g. the global batch count.
    step: NonNegativeInt
    metrics: Metrics
    # Keyed by layer name; values are measured activations, possibly sampled.
    selected_activations: Annotated[
        dict[
            Identifier,
            Annotated[list[FiniteFloat], Field(max_length=MAX_VALUES_PER_LAYER)],
        ],
        Field(max_length=MAX_ACTIVATION_LAYERS),
    ] = Field(default_factory=dict)
    health: StepHealth | None = None


class EventBatch(_Strict):
    # An empty batch is a heartbeat: it keeps an idle run from timing out.
    events: Annotated[list[LoggedStep], Field(max_length=MAX_EVENTS_PER_REQUEST)]


class RunFinish(_Strict):
    # "cancelled" when the script stopped because of a stop command.
    status: Literal["completed", "failed", "cancelled"]
    message: Annotated[str, Field(min_length=1, max_length=500)] | None = None


class RunInfo(BaseModel):
    run_id: str
    name: str
    model_id: str | None
    status: RunStatus
    total_epochs: int
    epoch: int
    step: int
    steps_logged: int
    # Unix seconds.
    created_at: float
    last_event_at: float
    controls: list[CommandKind]
    # As last reported by the script (an applied pause/resume, the optimizer).
    paused: bool
    learning_rate: float | None
    pending_commands: int
    alerts: int


class CommandRequest(_Strict):
    """Issued from the dashboard; the script applies it at its next run.poll()."""

    kind: CommandKind
    # Required for set_lr: the new learning rate.
    value: Annotated[NonNegativeFloat, Field(gt=0, le=MAX_LEARNING_RATE)] | None = None


class CommandAck(_Strict):
    """The script's answer to one command."""

    status: Literal["applied", "rejected"]
    message: Message | None = None
    # The script's state after applying it.
    paused: bool | None = None
    learning_rate: NonNegativeFloat | None = None


class CommandList(BaseModel):
    commands: list[RunCommand]


class RunCreated(BaseModel):
    run: RunInfo
    # Required as "Authorization: Bearer <token>" to log to this run. It is
    # returned only once and never stored in plain text.
    write_token: str


class RunList(BaseModel):
    runs: list[RunInfo]


class IngestResult(BaseModel):
    accepted: int
    steps_logged: int
    # Pending commands, so a script that logs often needs no separate polling.
    commands: list[RunCommand] = Field(default_factory=list)
