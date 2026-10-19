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

Metrics = Annotated[dict[str, FiniteFloat], Field(max_length=MAX_METRICS)]
NonNegativeInt = Annotated[int, Field(ge=0)]


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


TrainingEvent = Annotated[
    TrainingStarted
    | TrainingStep
    | TrainingCompleted
    | TrainingCancelled
    | TrainingFailed,
    Field(discriminator="event_type"),
]

training_event_adapter: TypeAdapter[TrainingEvent] = TypeAdapter(TrainingEvent)
