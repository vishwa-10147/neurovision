# Inspecting and running your own models (ONNX)

The **Model Explorer** opens trained models from real projects. It shows
their computation graph, operators, tensor shapes, and weights, and it
can **run** them on inputs you choose, showing the values every operator
actually produces. Runs happen in ONNX Runtime inside a separate,
time-limited worker process on the server; uploading never runs a model
by itself.

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

## Running a model

The **Run the model** panel lists each input the model declares, with
its type and shape:

-   Fixed dimensions (e.g. `3 × 32 × 32`) are locked. Named or unknown
    ones (e.g. `batch`) default to 1 and can be changed.
-   Values can be random (seeded, so repeatable), zeros, ones, evenly
    spaced from −1 to 1, or typed in (up to 4,096 values).

After **Run** you get:

-   The execution time inside ONNX Runtime. On the first run, notes also
    report that the worker started and how long loading the model took.
-   Each output's shape and first values. For `[batch, classes]` outputs,
    the top 5 values of the first row (indices only; the file has no
    class names).
-   **Operator colors** in the 2D graph: each block is shaded by the
    root-mean-square size of its output in this run, scaled across
    operators.
-   In the inspector, each operator's real output (and each link's
    tensor): shape, min/max/mean/std, and the first values.

Verified with the PyTorch export above: for the same seeded input, the
top logits NeuroVision reported matched PyTorch's own output exactly to
four decimals.

### Limits

| Limit | Value |
|---|---|
| Time per run | 10 s (`NEUROVISION_INFERENCE_TIMEOUT_SECONDS`); the worker is stopped and restarted |
| Worker memory | 2 GiB on Linux/macOS (`NEUROVISION_INFERENCE_MAX_MEMORY_BYTES`); **not enforced on Windows** |
| Input size | 1,000,000 values in total; each dimension 1–100,000 |
| Intermediates | All operators up to 200 operators; above that only those you request (API `capture`) |
| Values returned per tensor | First 4,096, plus a summary of all of them |
| Concurrency | One run at a time (others get "try again") |

A model cannot run here if it uses non-standard operators (other than
ONNX Runtime's `com.microsoft` ones), keeps weights in files that were
not uploaded, or has input types other than float, double, float16,
int8/16/32/64, uint8, or bool. The panel lists the reasons; inspection
still works.

## Supported and unsupported cases

| Case | Behavior |
|---|---|
| Standard ONNX operators (any opset) | Imported and drawn normally. |
| Custom or vendor operators (e.g. `com.microsoft` domains) | Kept in the graph with a dashed outline, listed under "Unsupported operators", and named in a warning. Nothing is dropped. Models with custom operators cannot be run; `com.microsoft` operators can, because ONNX Runtime implements them. |
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
