"""Running imported models with ONNX Runtime: inputs, limits, and results."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from app.schemas.graph import Shape
from app.schemas.training import TensorSummary

MAX_INPUT_ELEMENTS = 1_000_000
MAX_DIMENSION = 100_000
MAX_RANK = 8
# Above this many operators, intermediates are captured only on request.
MAX_AUTO_CAPTURE = 200
MAX_CAPTURE = 256

# ONNX element types that inputs may have, and the NumPy dtype used for each.
SUPPORTED_INPUT_DTYPES = {
    "float": "float32",
    "double": "float64",
    "float16": "float16",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "bool": "bool",
}


class InputSpec(BaseModel):
    """A graph input the model expects, as declared in the file."""

    name: str
    node_id: str
    dtype: str
    # Declared shape; symbolic or unknown dims are any size >= 1.
    shape: Shape


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: Annotated[
        list[Annotated[int, Field(ge=1, le=MAX_DIMENSION)]], Field(max_length=MAX_RANK)
    ]
    # Row-major values; omit to generate them with `fill`.
    values: (
        Annotated[list[FiniteFloat], Field(max_length=MAX_INPUT_ELEMENTS)] | None
    ) = None
    fill: Literal["random", "zeros", "ones", "range"] = "random"
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: Annotated[dict[str, RunInput], Field(max_length=16)]
    # Node ids whose outputs to return; omit for all operators in small graphs.
    capture: Annotated[list[str], Field(max_length=MAX_CAPTURE)] | None = None


class TensorValues(BaseModel):
    name: str
    # The operator (or input) that produced this tensor, when known.
    node_id: str | None
    shape: list[int]
    dtype: str
    count: int
    summary: TensorSummary | None
    non_finite_count: int
    values: list[float | None]
    truncated: bool


class RunInputShape(BaseModel):
    name: str
    node_id: str
    shape: list[int]
    dtype: str


class RunResponse(BaseModel):
    model_id: str
    # The inputs as actually fed (shapes after defaults), for map views.
    inputs: list[RunInputShape] = Field(default_factory=list)
    # Identifies this run's kept tensors for 2D map views.
    run_id: str
    # False when the tensors were too large to keep (map views unavailable).
    retained: bool
    runtime: str
    # Time spent executing the model in ONNX Runtime (excludes loading it).
    elapsed_ms: float
    outputs: list[TensorValues]
    intermediates: list[TensorValues]
    notes: list[str] = Field(default_factory=list)


class TensorViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Annotated[
        str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    ]
    node_id: Annotated[str, Field(min_length=1, max_length=64)]
    # Which of the node's outputs (0 for most operators).
    output: Annotated[int, Field(ge=0, le=64)] = 0
    # Indices for the leading dimensions, e.g. [batch] for [N, C, H, W] grids.
    fixed: Annotated[list[Annotated[int, Field(ge=0)]], Field(max_length=MAX_RANK)] = (
        Field(default_factory=list)
    )


Cells = list[list[float | None]]


class _ViewBase(BaseModel):
    model_id: str
    run_id: str
    node_id: str
    tensor: str
    shape: list[int]
    fixed: list[int]
    # Finite min/max over the viewed values (null when there are none).
    min: float | None
    max: float | None
    non_finite_count: int


class TensorMapResponse(_ViewBase):
    """One 2D map: the last two dimensions at the `fixed` leading indices."""

    rows: int
    cols: int
    # Block size used to shrink the map to at most 65,536 cells (1 = exact).
    stride: int
    values: Cells


class GridItem(BaseModel):
    index: int
    stride: int
    values: Cells
    min: float | None
    max: float | None
    non_finite_count: int


class TensorGridResponse(_ViewBase):
    """One thumbnail per index along `axis` (channels, attention heads)."""

    axis: int
    # How many maps the axis has; at most 256 are returned.
    items_total: int
    rows: int
    cols: int
    items: list[GridItem]
