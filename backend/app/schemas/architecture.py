"""Editable configuration for educational dense (fully connected) networks.

Limits keep neuron-level 2D rendering and future CPU training responsive.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.graph import ModelGraph

MAX_INPUT_UNITS = 16
MAX_HIDDEN_UNITS = 32
MAX_OUTPUT_UNITS = 10
MAX_HIDDEN_LAYERS = 6

Activation = Literal["relu", "sigmoid", "tanh"]


class HiddenLayerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    units: Annotated[int, Field(ge=1, le=MAX_HIDDEN_UNITS)]
    activation: Activation


class DenseNetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=80)]
    input_units: Annotated[int, Field(ge=1, le=MAX_INPUT_UNITS)]
    hidden_layers: Annotated[
        list[HiddenLayerConfig], Field(max_length=MAX_HIDDEN_LAYERS)
    ]
    output_units: Annotated[int, Field(ge=1, le=MAX_OUTPUT_UNITS)]
    # None means the output layer emits raw values (logits).
    output_activation: Activation | None = None


class EducationalModelCreated(BaseModel):
    model_id: str
    graph: ModelGraph
