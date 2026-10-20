"""External runs: training scripts outside NeuroVision that stream their own
metrics and activations through POST /api/v1/runs/{run_id}/events.

Logged steps use the same bounds as training.step events, so the existing
event stream and UI can follow them unchanged.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from app.schemas.events import MAX_ACTIVATION_LAYERS, MAX_VALUES_PER_LAYER, Metrics
from app.schemas.graph import Identifier

MAX_EVENTS_PER_REQUEST = 32
# Steps a run may log in total; the replay history stays bounded too.
MAX_STEPS_PER_RUN = 10_000
MAX_TOTAL_EPOCHS = 1_000_000

RunStatus = Literal["running", "completed", "failed"]
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


class EventBatch(_Strict):
    # An empty batch is a heartbeat: it keeps an idle run from timing out.
    events: Annotated[list[LoggedStep], Field(max_length=MAX_EVENTS_PER_REQUEST)]


class RunFinish(_Strict):
    status: Literal["completed", "failed"]
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
