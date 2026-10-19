"""Request and response models for forward passes, tensors, and training."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from app.ml.datasets import DatasetName
from app.schemas.events import MAX_ACTIVATION_LAYERS

MAX_FORWARD_ROWS = 64
MAX_INPUT_MAGNITUDE = 1e6
MAX_TENSOR_VALUES = 4096

InputValue = Annotated[
    FiniteFloat, Field(ge=-MAX_INPUT_MAGNITUDE, le=MAX_INPUT_MAGNITUDE)
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForwardRequest(_Strict):
    # One row per example; each row has one value per input unit.
    inputs: Annotated[
        list[list[InputValue]], Field(min_length=1, max_length=MAX_FORWARD_ROWS)
    ]
    # Graph node ids to return values for; omit for every layer.
    capture: Annotated[list[str], Field(max_length=16)] | None = None


class LayerValues(BaseModel):
    # W·x + b per row (null for the input layer); post is after the activation.
    pre: list[list[float]] | None
    post: list[list[float]]


class ForwardResponse(BaseModel):
    model_id: str
    # Output layer values after its configured activation.
    outputs: list[list[float]]
    # Output layer values before the activation; training uses these.
    logits: list[list[float]]
    predictions: list[int]
    activations: dict[str, LayerValues]


class TensorSummary(BaseModel):
    min: float
    max: float
    mean: float
    std: float


class TensorResponse(BaseModel):
    tensor_id: str
    shape: list[int]
    dtype: str
    count: int
    summary: TensorSummary
    offset: int
    # Row-major values from `offset`; a weight is [out_units][in_units], so
    # weight[j][i] connects source unit i to target unit j.
    values: list[float]
    truncated: bool


class TrainingConfig(_Strict):
    dataset: DatasetName = "moons"
    samples: Annotated[int, Field(ge=50, le=2000)] = 400
    noise: Annotated[float, Field(ge=0, le=1)] = 0.1
    validation_split: Annotated[float, Field(ge=0.1, le=0.5)] = 0.2
    epochs: Annotated[int, Field(ge=1, le=500)] = 100
    batch_size: Annotated[int, Field(ge=1, le=512)] = 32
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.03
    optimizer: Literal["sgd", "adam"] = "adam"
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0
    # Layers whose activations on a fixed validation example are streamed.
    capture_layers: Annotated[list[str], Field(max_length=MAX_ACTIVATION_LAYERS)] = (
        Field(default_factory=list)
    )
    # Pause after each epoch so training can be watched; 0 runs at full speed.
    epoch_delay_ms: Annotated[int, Field(ge=0, le=500)] = 0


JobStatus = Literal["running", "completed", "cancelled", "failed"]


class TrainingJobInfo(BaseModel):
    job_id: str
    model_id: str
    status: JobStatus
    epoch: int
    total_epochs: int
    config: TrainingConfig


class EpochMetrics(BaseModel):
    epoch: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float


class MetricsResponse(BaseModel):
    model_id: str
    job: TrainingJobInfo | None
    history: list[EpochMetrics]
