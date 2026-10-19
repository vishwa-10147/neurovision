"""Small sample ONNX models, built in memory, for trying the Model Explorer.

They go through the same importer as uploads. Weights are seeded random
values: the samples show structure and tensors, not trained behavior.
"""

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

from app.schemas.imports import SampleModel

SAMPLES = [
    SampleModel(
        name="mlp",
        title="Tiny MLP",
        description="Gemm → ReLU → Gemm → Softmax with a dynamic batch size.",
    ),
    SampleModel(
        name="cnn",
        title="Small CNN",
        description="Two Conv/ReLU/MaxPool stages, Flatten, and a Gemm classifier for 28×28 images.",
    ),
    SampleModel(
        name="residual",
        title="Residual block",
        description="Conv → ReLU → Conv with a skip connection joined by Add.",
    ),
    SampleModel(
        name="custom-op",
        title="Custom operator",
        description="An MLP with a non-standard operator, to show how unknown operators appear.",
    ),
]

_OPSET = [helper.make_opsetid("", 17)]


def _weights(
    name: str, shape: tuple[int, ...], rng: np.random.Generator
) -> TensorProto:
    return numpy_helper.from_array(
        (rng.standard_normal(shape) * 0.1).astype(np.float32), name
    )


def _model(graph_name: str, nodes, inputs, outputs, inits, opsets=_OPSET) -> ModelProto:  # type: ignore[no-untyped-def]
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, inits)
    model = helper.make_model(
        graph, opset_imports=opsets, producer_name="neurovision-samples"
    )
    model.ir_version = 8
    return model


def build_sample(name: str) -> ModelProto | None:
    rng = np.random.default_rng(0)
    f32 = TensorProto.FLOAT
    if name in ("mlp", "custom-op"):
        nodes = [
            helper.make_node("Gemm", ["x", "w1", "b1"], ["h1"], name="fc1", transB=1),
            helper.make_node("Relu", ["h1"], ["a1"], name="relu1"),
        ]
        last = "a1"
        opsets = _OPSET
        if name == "custom-op":
            nodes.append(
                helper.make_node(
                    "Scale",
                    ["a1"],
                    ["s1"],
                    name="custom_scale",
                    domain="com.example",
                    factor=2.0,
                )
            )
            last = "s1"
            opsets = [*_OPSET, helper.make_opsetid("com.example", 1)]
        nodes += [
            helper.make_node(
                "Gemm", [last, "w2", "b2"], ["logits"], name="fc2", transB=1
            ),
            helper.make_node("Softmax", ["logits"], ["probs"], name="softmax", axis=1),
        ]
        inits = [
            _weights("w1", (8, 4), rng),
            _weights("b1", (8,), rng),
            _weights("w2", (3, 8), rng),
            _weights("b2", (3,), rng),
        ]
        return _model(
            "tiny_mlp" if name == "mlp" else "mlp_with_custom_op",
            nodes,
            [helper.make_tensor_value_info("x", f32, ["batch", 4])],
            [helper.make_tensor_value_info("probs", f32, ["batch", 3])],
            inits,
            opsets,
        )
    if name == "cnn":
        nodes = [
            helper.make_node(
                "Conv",
                ["image", "c1w", "c1b"],
                ["c1"],
                name="conv1",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
            helper.make_node(
                "MaxPool",
                ["r1"],
                ["p1"],
                name="pool1",
                kernel_shape=[2, 2],
                strides=[2, 2],
            ),
            helper.make_node(
                "Conv",
                ["p1", "c2w", "c2b"],
                ["c2"],
                name="conv2",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Relu", ["c2"], ["r2"], name="relu2"),
            helper.make_node(
                "MaxPool",
                ["r2"],
                ["p2"],
                name="pool2",
                kernel_shape=[2, 2],
                strides=[2, 2],
            ),
            helper.make_node("Flatten", ["p2"], ["flat"], name="flatten", axis=1),
            helper.make_node(
                "Gemm", ["flat", "fw", "fb"], ["logits"], name="classifier", transB=1
            ),
        ]
        inits = [
            _weights("c1w", (8, 1, 3, 3), rng),
            _weights("c1b", (8,), rng),
            _weights("c2w", (16, 8, 3, 3), rng),
            _weights("c2b", (16,), rng),
            _weights("fw", (10, 16 * 7 * 7), rng),
            _weights("fb", (10,), rng),
        ]
        return _model(
            "small_cnn",
            nodes,
            [helper.make_tensor_value_info("image", f32, ["N", 1, 28, 28])],
            [helper.make_tensor_value_info("logits", f32, ["N", 10])],
            inits,
        )
    if name == "residual":
        nodes = [
            helper.make_node(
                "Conv",
                ["x", "w1"],
                ["c1"],
                name="conv_a",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Relu", ["c1"], ["r1"], name="relu_a"),
            helper.make_node(
                "Conv",
                ["r1", "w2"],
                ["c2"],
                name="conv_b",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Add", ["c2", "x"], ["sum"], name="skip_add"),
            helper.make_node("Relu", ["sum"], ["y"], name="relu_out"),
        ]
        inits = [
            _weights("w1", (16, 16, 3, 3), rng),
            _weights("w2", (16, 16, 3, 3), rng),
        ]
        return _model(
            "residual_block",
            nodes,
            [helper.make_tensor_value_info("x", f32, ["N", 16, "H", "W"])],
            [helper.make_tensor_value_info("y", f32, ["N", 16, "H", "W"])],
            inits,
        )
    return None
