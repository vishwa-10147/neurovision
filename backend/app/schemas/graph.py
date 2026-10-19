"""Normalized model-graph contract shared by importers, trainers, and renderers.

The graph describes semantics only: no layout coordinates and no tensor data.
Large tensors are referenced by name and served by bounded tensor endpoints.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "1.0"

Identifier = Annotated[str, Field(min_length=1, max_length=256)]

# A known size, a symbolic size such as "batch", or null when unknown.
# Shapes are never invented: unknown dimensions stay null.
Dimension = Annotated[int, Field(ge=0)] | Annotated[str, Field(min_length=1)] | None

# A tensor shape, or null when the rank itself is unknown.
Shape = list[Dimension] | None

# Source-specific data that does not normalize cleanly, keyed by namespace
# (for example "onnx" or "pytorch").
NamespacedMetadata = dict[str, dict[str, JsonValue]]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Capabilities(StrictModel):
    """What the current model and runtime actually support."""

    graph: bool
    inference: bool
    training: bool
    activations: bool
    gradients: bool
    attention: bool


class ModelInfo(StrictModel):
    id: Identifier
    name: str
    source_format: Literal["educational", "onnx"]
    capabilities: Capabilities
    metadata: NamespacedMetadata = Field(default_factory=dict)


class ParameterSummary(StrictModel):
    parameter_count: Annotated[int, Field(ge=0)]
    trainable: bool | None = None


class GraphNode(StrictModel):
    id: Identifier
    name: str
    op_type: Annotated[str, Field(min_length=1)]
    input_tensors: list[str] = Field(default_factory=list)
    output_tensors: list[str] = Field(default_factory=list)
    input_shapes: list[Shape] = Field(default_factory=list)
    output_shapes: list[Shape] = Field(default_factory=list)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    parameter_summary: ParameterSummary | None = None
    # True when the operator is preserved for inspection but not understood.
    unsupported: bool = False
    metadata: NamespacedMetadata = Field(default_factory=dict)


class GraphEdge(StrictModel):
    id: Identifier
    source: Identifier
    target: Identifier
    tensor_name: str | None = None


class ModelGraph(StrictModel):
    """Normalized model graph, schema version 1.0.

    Integrity rules not expressible in JSON Schema (enforced by the backend and
    frontend validators): node ids and edge ids are unique; every edge source
    and target is an existing node id; non-empty input/output shape lists have
    one entry per input/output tensor.
    """

    schema_version: Literal["1.0"]
    model: ModelInfo
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_integrity(self) -> Self:
        node_ids = [node.id for node in self.nodes]
        if duplicate := _first_duplicate(node_ids):
            raise ValueError(f"duplicate node id '{duplicate}'")
        if duplicate := _first_duplicate([edge.id for edge in self.edges]):
            raise ValueError(f"duplicate edge id '{duplicate}'")
        known = set(node_ids)
        for edge in self.edges:
            for end in (edge.source, edge.target):
                if end not in known:
                    raise ValueError(
                        f"edge '{edge.id}' references unknown node '{end}'"
                    )
        for node in self.nodes:
            if node.input_shapes and len(node.input_shapes) != len(node.input_tensors):
                raise ValueError(
                    f"node '{node.id}' has {len(node.input_shapes)} input shapes "
                    f"for {len(node.input_tensors)} input tensors"
                )
            if node.output_shapes and len(node.output_shapes) != len(
                node.output_tensors
            ):
                raise ValueError(
                    f"node '{node.id}' has {len(node.output_shapes)} output shapes "
                    f"for {len(node.output_tensors)} output tensors"
                )
        return self


def _first_duplicate(values: list[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
