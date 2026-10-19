"""Build the normalized graph for an educational dense network.

Graph construction is pure arithmetic over the configuration. Each educational
graph is backed by a PyTorch network (app/ml/educational_network.py), so it
supports inference, training, activations, and gradients.
"""

from app.schemas.architecture import DenseNetworkConfig
from app.schemas.graph import (
    SCHEMA_VERSION,
    Capabilities,
    GraphEdge,
    GraphNode,
    ModelGraph,
    ModelInfo,
    ParameterSummary,
)

EDUCATIONAL_CAPABILITIES = Capabilities(
    graph=True,
    inference=True,
    training=True,
    activations=True,
    gradients=True,
    attention=False,
)


def dense_parameter_count(in_features: int, out_features: int) -> int:
    """Weights plus one bias per output unit."""
    return in_features * out_features + out_features


def build_dense_graph(config: DenseNetworkConfig, model_id: str) -> ModelGraph:
    nodes = [
        GraphNode(
            id="input",
            name="Input",
            op_type="Input",
            output_tensors=["x"],
            output_shapes=[["batch", config.input_units]],
        )
    ]

    layers = [(layer.units, layer.activation) for layer in config.hidden_layers]
    layers.append((config.output_units, config.output_activation))

    previous_id, previous_tensor, previous_units = "input", "x", config.input_units
    edges: list[GraphEdge] = []
    total = 0
    for index, (units, activation) in enumerate(layers, start=1):
        is_output = index == len(layers)
        node_id = "output" if is_output else f"hidden_{index}"
        tensor = "logits" if is_output else f"h{index}"
        count = dense_parameter_count(previous_units, units)
        total += count
        nodes.append(
            GraphNode(
                id=node_id,
                name="Output Layer" if is_output else f"Hidden Layer {index}",
                op_type="Linear",
                input_tensors=[previous_tensor],
                output_tensors=[tensor],
                input_shapes=[["batch", previous_units]],
                output_shapes=[["batch", units]],
                attributes={
                    "in_features": previous_units,
                    "out_features": units,
                    "activation": activation,
                },
                parameter_summary=ParameterSummary(
                    parameter_count=count, trainable=True
                ),
            )
        )
        edges.append(
            GraphEdge(
                id=f"e_{previous_id}_{node_id}",
                source=previous_id,
                target=node_id,
                tensor_name=previous_tensor,
            )
        )
        previous_id, previous_tensor, previous_units = node_id, tensor, units

    return ModelGraph(
        schema_version=SCHEMA_VERSION,
        model=ModelInfo(
            id=model_id,
            name=config.name,
            source_format="educational",
            capabilities=EDUCATIONAL_CAPABILITIES,
            metadata={
                "educational": {
                    "parameter_count": total,
                    "config": config.model_dump(mode="json"),
                }
            },
        ),
        nodes=nodes,
        edges=edges,
    )
