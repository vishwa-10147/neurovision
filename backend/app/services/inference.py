"""Forward passes and bounded tensor access for educational models."""

import re

import torch

from app.core.errors import invalid, not_found
from app.ml.educational_network import LayerTrace
from app.schemas.training import (
    ForwardRequest,
    ForwardResponse,
    LayerValues,
    TensorResponse,
)
from app.services.model_repository import ModelRecord
from app.services.tensors import tensor_response

TENSOR_ID = re.compile(
    r"^(?P<node>[a-z0-9_]+)\.(?P<kind>weight|bias)(?P<grad>\.grad)?$"
)


def run_forward(record: ModelRecord, request: ForwardRequest) -> ForwardResponse:
    network = record.network
    for row, values in enumerate(request.inputs):
        if len(values) != network.input_units:
            raise invalid(
                f"body.inputs.{row}",
                f"Expected {network.input_units} values (one per input unit), "
                f"got {len(values)}.",
            )
    known = {"input", *network.node_ids}
    for i, node_id in enumerate(request.capture or []):
        if node_id not in known:
            raise invalid(f"body.capture.{i}", f"Unknown layer '{node_id}'.")

    x = torch.tensor(request.inputs, dtype=torch.float32)
    with record.lock, torch.no_grad():
        traces = network.trace(x)
    output = traces["output"]
    assert output.pre is not None
    logits = output.pre
    predictions = (
        (logits.squeeze(1) > 0).long()
        if network.output_units == 1
        else logits.argmax(dim=1)
    )
    wanted = request.capture if request.capture is not None else list(traces)
    return ForwardResponse(
        model_id=record.model_id,
        outputs=output.post.tolist(),
        logits=logits.tolist(),
        predictions=predictions.tolist(),
        activations={node_id: _values(traces[node_id]) for node_id in wanted},
    )


def _values(trace: LayerTrace) -> LayerValues:
    pre = None if trace.pre is None else trace.pre.tolist()
    return LayerValues(pre=pre, post=trace.post.tolist())


def read_tensor(
    record: ModelRecord, tensor_id: str, offset: int, limit: int
) -> TensorResponse:
    match = TENSOR_ID.match(tensor_id)
    parameters = record.network.parameter_tensors()
    if not match or f"{match['node']}.{match['kind']}" not in parameters:
        raise not_found(
            f"Tensor '{tensor_id}' does not exist. Use ids like 'hidden_1.weight', "
            "'output.bias', or 'hidden_1.weight.grad'."
        )
    with record.lock:
        if match["grad"]:
            tensor = record.gradients.get(tensor_id)
            if tensor is None:
                raise not_found(
                    f"No gradients for '{tensor_id}' yet. Train the model first."
                )
        else:
            tensor = parameters[tensor_id]
        flat = tensor.detach().flatten().numpy().copy()

    return tensor_response(
        tensor_id,
        list(tensor.shape),
        str(tensor.dtype).removeprefix("torch."),
        flat,
        offset,
        limit,
    )
