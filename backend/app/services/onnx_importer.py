"""Parse ONNX models into the normalized graph, without executing them.

Uploaded files are untrusted. They are parsed from bytes in memory only:
external data references are never followed (nothing is read from disk),
the model is never run, and every string copied from the file is truncated.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import onnx
from onnx import AttributeProto, ModelProto, TensorProto, ValueInfoProto

from app.schemas.graph import (
    SCHEMA_VERSION,
    Capabilities,
    Dimension,
    GraphEdge,
    GraphNode,
    ModelGraph,
    ModelInfo,
    ParameterSummary,
    Shape,
)
from app.schemas.imports import ImportReport, OpsetImport, UnsupportedOperator

MAX_OPERATORS = 5000
MAX_TEXT = 200
MAX_LIST = 64
MAX_WARNINGS = 50
# Names shape inference makes up for dimensions it cannot determine.
_PLACEHOLDER = re.compile(r"^unk__\d+$")

# Imported graphs can be inspected; running them arrives with ONNX Runtime.
INSPECTION_ONLY = Capabilities(
    graph=True,
    inference=False,
    training=False,
    activations=False,
    gradients=False,
    attention=False,
)


class OnnxImportError(ValueError):
    """The upload is not an importable ONNX model; the message is client-safe."""


@dataclass
class ImportResult:
    graph: ModelGraph
    initializers: dict[str, TensorProto]
    report: ImportReport


def _text(value: str | bytes, limit: int = MAX_TEXT) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _dtype(data_type: int) -> str:
    try:
        return TensorProto.DataType.Name(data_type).lower()
    except ValueError:
        return "unknown"


def is_external(tensor: TensorProto) -> bool:
    return tensor.data_location == TensorProto.EXTERNAL


def _shape(info: ValueInfoProto) -> Shape:
    """Declared or inferred shape; unknown dimensions stay null, never invented."""
    tensor_type = info.type.tensor_type
    if not info.type.HasField("tensor_type") or not tensor_type.HasField("shape"):
        return None
    dims: list[Dimension] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value >= 0:
            dims.append(dim.dim_value)
        elif (
            dim.HasField("dim_param")
            and dim.dim_param
            and not _PLACEHOLDER.match(dim.dim_param)
        ):
            dims.append(_text(dim.dim_param, 64))
        else:
            dims.append(None)
    return dims


def _attribute(attr: AttributeProto, warnings: list[str], node_label: str) -> Any:
    """JSON-safe, bounded view of an attribute value."""
    kind = attr.type
    if kind == AttributeProto.FLOAT:
        return attr.f if math.isfinite(attr.f) else str(attr.f)
    if kind == AttributeProto.INT:
        return attr.i
    if kind == AttributeProto.STRING:
        return _text(attr.s)
    if kind == AttributeProto.FLOATS:
        return [f if math.isfinite(f) else str(f) for f in attr.floats[:MAX_LIST]]
    if kind == AttributeProto.INTS:
        return list(attr.ints[:MAX_LIST])
    if kind == AttributeProto.STRINGS:
        return [_text(s) for s in attr.strings[:MAX_LIST]]
    if kind == AttributeProto.TENSOR:
        return {
            "tensor": {"shape": list(attr.t.dims), "dtype": _dtype(attr.t.data_type)}
        }
    if kind in (AttributeProto.GRAPH, AttributeProto.GRAPHS):
        graphs = [attr.g] if kind == AttributeProto.GRAPH else list(attr.graphs)
        warnings.append(
            f"{node_label} contains a subgraph ('{attr.name}') that is not expanded in this view."
        )
        return {
            "subgraphs": [
                {"name": _text(g.name), "operators": len(g.node)} for g in graphs
            ]
        }
    return {
        "unsupported_attribute_type": AttributeProto.AttributeType.Name(kind).lower()
    }


def _parse(data: bytes) -> ModelProto:
    if not data:
        raise OnnxImportError("The file is empty.")
    try:
        # Bytes only: external data paths inside the file are never opened.
        model = onnx.load_model_from_string(data)
    except Exception as error:  # Protobuf raises several error types.
        raise OnnxImportError(
            "The file is not a valid ONNX model; it could not be parsed."
        ) from error
    # Arbitrary bytes can decode as an empty protobuf message; require a model.
    if model.ir_version <= 0 or not model.HasField("graph"):
        raise OnnxImportError(
            "The file is not a valid ONNX model; it has no IR version or graph."
        )
    if len(model.graph.node) > MAX_OPERATORS:
        raise OnnxImportError(
            f"The model has {len(model.graph.node):,} operators; this version can inspect at most "
            f"{MAX_OPERATORS:,}."
        )
    return model


def import_onnx(data: bytes, model_id: str, file_name: str) -> ImportResult:
    model = _parse(data)
    graph = model.graph
    warnings: list[str] = []

    initializers = {t.name: t for t in graph.initializer}
    external = [t for t in graph.initializer if is_external(t)]
    if external:
        warnings.append(
            f"{len(external)} tensor(s) store their values in external files that were not "
            "uploaded; their shapes are shown but their values are unavailable."
        )
        # The checker may look for external files on disk; never let it.
        warnings.append(
            "The ONNX checker was skipped because the model references external files."
        )
    else:
        try:
            onnx.checker.check_model(model, full_check=False)
        except Exception as error:
            first_line = (
                str(error).strip().splitlines()[0]
                if str(error).strip()
                else "unknown error"
            )
            warnings.append(
                f"The model does not pass the ONNX checker: {_text(first_line)}"
            )

    shapes: dict[str, Shape] = {}
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model, check_type=False, strict_mode=False
        )
        value_infos = list(inferred.graph.value_info)
    except Exception:
        value_infos = []
        warnings.append(
            "Shape inference failed; only shapes declared in the file are shown."
        )
    for info in [*graph.input, *graph.output, *value_infos]:
        if info.name and shapes.get(info.name) is None:
            shapes[info.name] = _shape(info)
    for tensor in graph.initializer:
        shapes[tensor.name] = list(tensor.dims)

    tensor_ids = {name: f"t{i}" for i, name in enumerate(initializers)}
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    producer: dict[str, str] = {}

    # Graph inputs that are not weights become Input nodes.
    for i, info in enumerate(i for i in graph.input if i.name not in initializers):
        node_id = f"input_{i}"
        producer[info.name] = node_id
        nodes.append(
            GraphNode(
                id=node_id,
                name=_text(info.name),
                op_type="Input",
                output_tensors=[_text(info.name)],
                output_shapes=[shapes.get(info.name)],
            )
        )

    unsupported: Counter[tuple[str, str]] = Counter()
    for i, op in enumerate(graph.node):
        node_id = f"node_{i}"
        label = (
            f"Operator '{_text(op.name or op.op_type, 80)}' ({_text(op.op_type, 80)})"
        )
        known = onnx.defs.has(op.op_type, op.domain)
        if not known:
            unsupported[(_text(op.op_type, 80), op.domain or "ai.onnx")] += 1
        inputs = [name for name in op.input if name]
        weights = [name for name in inputs if name in initializers]
        parameter_count = sum(math.prod(initializers[name].dims) for name in weights)
        nodes.append(
            GraphNode(
                id=node_id,
                name=_text(op.name or f"{op.op_type}_{i}"),
                op_type=_text(op.op_type, 80),
                input_tensors=[_text(n) for n in inputs],
                output_tensors=[_text(n) for n in op.output if n],
                input_shapes=[shapes.get(n) for n in inputs],
                output_shapes=[shapes.get(n) for n in op.output if n],
                attributes={
                    a.name: _attribute(a, warnings, label)
                    for a in op.attribute[:MAX_LIST]
                },
                parameter_summary=ParameterSummary(parameter_count=parameter_count)
                if weights
                else None,
                unsupported=not known,
                metadata={
                    "onnx": {
                        "domain": op.domain or "ai.onnx",
                        "initializers": [
                            {
                                "tensor_id": tensor_ids[name],
                                "name": _text(name),
                                "shape": list(initializers[name].dims),
                                "dtype": _dtype(initializers[name].data_type),
                                "external": is_external(initializers[name]),
                            }
                            for name in weights
                        ],
                        **({"doc": _text(op.doc_string, 500)} if op.doc_string else {}),
                    }
                },
            )
        )
        for name in inputs:
            if name in producer:
                edges.append(
                    GraphEdge(
                        id=f"e_{len(edges)}",
                        source=producer[name],
                        target=node_id,
                        tensor_name=_text(name),
                    )
                )
        for name in op.output:
            if name:
                producer[name] = node_id

    for i, info in enumerate(graph.output):
        node_id = f"output_{i}"
        nodes.append(
            GraphNode(
                id=node_id,
                name=_text(info.name),
                op_type="Output",
                input_tensors=[_text(info.name)],
                input_shapes=[shapes.get(info.name)],
            )
        )
        if info.name in producer:
            edges.append(
                GraphEdge(
                    id=f"e_{len(edges)}",
                    source=producer[info.name],
                    target=node_id,
                    tensor_name=_text(info.name),
                )
            )

    for (op_type, domain), count in sorted(unsupported.items()):
        warnings.append(
            f"Operator '{op_type}' (domain {domain}) is not a standard ONNX operator; "
            f"{count} node(s) are shown for inspection only."
        )
    if len(warnings) > MAX_WARNINGS:
        warnings = [
            *warnings[: MAX_WARNINGS - 1],
            f"…and {len(warnings) - MAX_WARNINGS + 1} more warnings.",
        ]

    opsets = [
        OpsetImport(domain=o.domain or "ai.onnx", version=o.version)
        for o in model.opset_import
    ]
    name = _text(graph.name) if graph.name else _text(file_name)
    model_graph = ModelGraph(
        schema_version=SCHEMA_VERSION,
        model=ModelInfo(
            id=model_id,
            name=name,
            source_format="onnx",
            capabilities=INSPECTION_ONLY,
            metadata={
                "onnx": {
                    "file_name": _text(file_name),
                    "ir_version": model.ir_version,
                    "producer_name": _text(model.producer_name),
                    "producer_version": _text(model.producer_version),
                    "model_version": model.model_version,
                    "opsets": [o.model_dump() for o in opsets],
                    "doc": _text(model.doc_string, 500),
                    "metadata_props": {
                        _text(p.key, 80): _text(p.value)
                        for p in list(model.metadata_props)[:20]
                    },
                }
            },
        ),
        nodes=nodes,
        edges=edges,
        warnings=warnings,
    )
    report = ImportReport(
        file_name=_text(file_name),
        file_bytes=len(data),
        operator_count=len(graph.node),
        edge_count=len(edges),
        initializer_count=len(initializers),
        parameter_count=sum(math.prod(t.dims) for t in graph.initializer),
        external_tensor_count=len(external),
        unsupported_operators=[
            UnsupportedOperator(op_type=op_type, domain=domain, count=count)
            for (op_type, domain), count in sorted(unsupported.items())
        ],
        opsets=opsets,
        warnings=warnings,
    )
    return ImportResult(
        graph=model_graph,
        initializers={
            tensor_ids[name]: tensor for name, tensor in initializers.items()
        },
        report=report,
    )
