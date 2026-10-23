"""Training events streamed over WS /api/v1/training/{job_id}/events.

Payloads are deliberately bounded: they carry metrics and a small sample of
selected activations, never the full model state.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter

from app.schemas.graph import Identifier

MAX_METRICS = 16
MAX_ACTIVATION_LAYERS = 8
MAX_VALUES_PER_LAYER = 256
# Health statistics (ADR 0008): parameter groups and activation layers per step.
MAX_HEALTH_LAYERS = 32
MAX_MESSAGE_LENGTH = 300

Metrics = Annotated[dict[str, FiniteFloat], Field(max_length=MAX_METRICS)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
Fraction = Annotated[FiniteFloat, Field(ge=0, le=1)]
Message = Annotated[str, Field(max_length=MAX_MESSAGE_LENGTH)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParameterHealth(_Strict):
    """One module that owns parameters, measured right after an optimizer step."""

    grad_norm: NonNegativeFloat | None = None
    weight_norm: NonNegativeFloat | None = None
    # ‖Δw‖ / ‖w‖ since the previous measurement; null on the first one.
    update_ratio: NonNegativeFloat | None = None
    parameters: NonNegativeInt = 0


class ActivationHealth(_Strict):
    """One activation module's output on the script's fixed probe batch."""

    kind: Literal["relu", "tanh", "sigmoid", "other"]
    mean: FiniteFloat
    std: NonNegativeFloat
    # Units (channels for conv outputs) that are <= 0 for every probe example.
    dead_fraction: Fraction | None = None
    # |y| > 0.99 for tanh; y < 0.01 or y > 0.99 for sigmoid.
    saturated_fraction: Fraction | None = None


class StepHealth(_Strict):
    """Training-health statistics measured by the script for one step."""

    grad_norm: NonNegativeFloat | None = None
    learning_rate: NonNegativeFloat | None = None
    nonfinite_grads: NonNegativeInt = 0
    # Metrics the script computed as NaN or infinite (their values are not sent).
    nonfinite_metrics: Annotated[list[Identifier], Field(max_length=MAX_METRICS)] = (
        Field(default_factory=list)
    )
    parameters: Annotated[
        dict[Identifier, ParameterHealth], Field(max_length=MAX_HEALTH_LAYERS)
    ] = Field(default_factory=dict)
    activations: Annotated[
        dict[Identifier, ActivationHealth], Field(max_length=MAX_HEALTH_LAYERS)
    ] = Field(default_factory=dict)


CommandKind = Literal["pause", "resume", "stop", "checkpoint", "set_lr"]
CommandStatus = Literal["pending", "applied", "rejected", "expired"]


class RunCommand(_Strict):
    command_id: Annotated[int, Field(ge=1)]
    kind: CommandKind
    # The new learning rate for set_lr; null otherwise.
    value: Annotated[FiniteFloat, Field(gt=0)] | None = None
    status: CommandStatus
    # Why it was rejected or expired, or what the script reported.
    message: Message | None = None
    # Unix seconds.
    created_at: float
    resolved_at: float | None = None


AlertKind = Literal[
    "non_finite_loss",
    "non_finite_gradients",
    "exploding_gradients",
    "vanishing_gradients",
    "update_ratio_high",
    "update_ratio_low",
    "dead_units",
    "saturated_units",
    "loss_spike",
    "loss_plateau",
    "overfitting",
    "alerts_suppressed",
]


class RunAlert(_Strict):
    alert_id: Annotated[int, Field(ge=1)]
    kind: AlertKind
    severity: Literal["info", "warning", "critical"]
    # The step whose statistics triggered it.
    step: NonNegativeInt
    # Parameter group or activation layer, when the alert is about one.
    layer: Identifier | None = None
    message: Message
    advice: Message


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    job_id: Identifier


class TrainingStarted(_EventBase):
    event_type: Literal["training.started"]
    model_id: Identifier
    total_epochs: Annotated[int, Field(ge=1)]


class TrainingStep(_EventBase):
    event_type: Literal["training.step"]
    epoch: NonNegativeInt
    step: NonNegativeInt
    metrics: Metrics
    # Keyed by graph node id; values are measured activations, possibly sampled.
    selected_activations: Annotated[
        dict[
            Identifier,
            Annotated[list[FiniteFloat], Field(max_length=MAX_VALUES_PER_LAYER)],
        ],
        Field(max_length=MAX_ACTIVATION_LAYERS),
    ] = Field(default_factory=dict)
    # External runs only: statistics measured by the script (ADR 0008).
    health: StepHealth | None = None


class TrainingCompleted(_EventBase):
    event_type: Literal["training.completed"]
    epoch: NonNegativeInt
    step: NonNegativeInt
    metrics: Metrics


class TrainingCancelled(_EventBase):
    event_type: Literal["training.cancelled"]
    step: NonNegativeInt


class TrainingFailed(_EventBase):
    event_type: Literal["training.failed"]
    message: Annotated[str, Field(max_length=500)]


class RunCommandEvent(_EventBase):
    """A command was issued to an external run, or its status changed."""

    event_type: Literal["run.command"]
    command: RunCommand
    # The run's state after this change, as reported by the script.
    paused: bool
    learning_rate: NonNegativeFloat | None = None


class RunAlertEvent(_EventBase):
    """The server's health rules flagged a problem in an external run."""

    event_type: Literal["run.alert"]
    alert: RunAlert


TrainingEvent = Annotated[
    TrainingStarted
    | TrainingStep
    | TrainingCompleted
    | TrainingCancelled
    | TrainingFailed
    | RunCommandEvent
    | RunAlertEvent,
    Field(discriminator="event_type"),
]

training_event_adapter: TypeAdapter[TrainingEvent] = TypeAdapter(TrainingEvent)
