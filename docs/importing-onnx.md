# Inspecting your own models (ONNX)

The **Model Explorer** opens trained models from real projects and shows
their computation graph, operators, tensor shapes, and weights. It
inspects models; it does not run them (running imported models is
planned for Milestone 6).

## Export a model from your project

### PyTorch

``` python
import torch

model.eval()                                   # your trained nn.Module
example = torch.randn(1, 3, 224, 224)          # one input of the right shape
torch.onnx.export(
    model, example, "my_model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
    dynamo=False,  # classic exporter; needs only the onnx package
)
```

Verified with PyTorch 2.14.1 on a small residual CNN: module names such
as `/block1/conv1/Conv` appear in the graph, the dynamic `batch`
dimension is kept, and both skip connections show up as two links into
each `Add`. In eval mode the exporter folds BatchNorm into the preceding
convolutions, so the graph shows the folded weights, which is what the
file contains.

### TensorFlow / Keras and scikit-learn

Use [`tf2onnx`](https://github.com/onnx/tensorflow-onnx) or
[`skl2onnx`](https://github.com/onnx/sklearn-onnx). The resulting files
import the same way. (These converters were not part of the verification
above.)

## What you see

-   **Graph:** one block per operator, labeled with its type (`Conv`,
    `Relu`, `Add`, …), plus blocks for graph inputs and outputs. Links are
    the tensors passed between operators. 2D and 3D views share the
    selection.
-   **Import report:** file, producer, opsets, operator / tensor /
    initializer counts, what you can and cannot do with the model, and
    warnings.
-   **Inspector:** the operator's name (your module path), domain,
    inputs and outputs with shapes, attributes, and its weight tensors.
    **Show values** loads a bounded summary (min, max, mean, std) and the
    first values; selecting a link shows the tensor's name and shape.

## Supported and unsupported cases

| Case | Behavior |
|---|---|
| Standard ONNX operators (any opset) | Imported and drawn normally. |
| Custom or vendor operators (e.g. `com.microsoft` domains) | Kept in the graph with a dashed outline, listed under "Unsupported operators", and named in a warning. Nothing is dropped. |
| Dynamic dimensions (`batch`, `N`, …) | Shown by name. Dimensions that shape inference cannot determine are shown as `?`, never invented. |
| Shape inference fails | Import continues with shapes declared in the file and a warning. |
| Model fails the ONNX checker | Import continues with a warning naming the first problem. |
| Weights in external data files | Shapes are shown; values are reported unavailable. The referenced paths are **never opened**, and the checker is skipped so it cannot touch the filesystem. |
| `If` / `Loop` / `Scan` subgraphs | Shown as one operator with a summary of each subgraph; subgraphs are not expanded (warning). |
| NaN or infinite weights | Counted, left out of the summary, and shown as `NaN/∞`. |
| String tensors | Shape shown; values cannot be summarized. |
| Files over 64 MB | Rejected with `413`; raise `NEUROVISION_MAX_UPLOAD_BYTES` if needed. |
| More than 5,000 operators | Rejected with a clear message. |
| Not ONNX (e.g. `.pt`, `.pb`, `.h5`) | Rejected; export to ONNX first. PyTorch checkpoints are never unpickled. |

## Limits of the views

-   The 3D view shows graphs up to 1,500 nodes, and labels every node
    only up to 60; larger graphs label just the selection. Long chains
    read best in 2D.
-   The 2D layout places operators by depth. Very wide graphs (many
    parallel branches) work but are not yet laid out to minimize
    crossings.
-   Imported models stay in memory only (lost on restart). Several large
    imports evict the oldest ones (512 MB budget).
