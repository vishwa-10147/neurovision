# ADR 0009: Showing a live run's network graph

Status: accepted

## Context

Live runs (ADR 0006, 0008) stream metrics, activations, health and system
figures from a user's own PyTorch script, but the dashboard showed them
only as charts and tables. The 2D and 3D views, the core of NeuroVision,
were available for imported ONNX models only, so users could not see the
network they were training change while it trained.

Constraints stay as before: NeuroVision never runs or imports user code,
needs no inbound connection to the user's machine, and keeps payloads
bounded.

## Decision

-   **The script exports, the server imports.** `start_run(model=...,
    example_input=...)` exports the model to ONNX in the script's process
    and uploads it to the existing, non-executing import endpoint
    (ADR 0004). The run is created with the returned `model_id`, which
    the API already accepted. Nothing new runs on the server.
-   **TorchScript exporter (`dynamo=False`).** It names every ONNX node
    after the module that produced it (`/layer1/layer1.0/relu/Relu`).
    The newer exporter does not keep this scope, and module names are
    what connects the stream to the graph. Its deprecation warning is
    silenced inside `export_onnx`, with a comment explaining why.
-   **Matching by module name, in the browser.** The SDK reports health
    and probe values per module name (`layer1.0.relu`). The frontend
    derives each node's module path from its name, folds the exporter's
    `_N` suffix for repeated calls, and places an activation on the last
    operator of its module (its output) and a gradient norm on every
    operator of the module that owns weights. Unmatched names are listed;
    unmatched operators stay uncolored. The page says matching is by name.
-   **Same renderers.** The live graph reuses `GraphView` with values
    supplied by the caller and a legend saying what they are; 3D blocks
    are now shaded on the same scale as 2D (this also colors imported
    models after a run in 3D).
-   **Failure is not fatal.** If export or upload fails, the run starts
    without a graph and the client logs a one-line reason (exporter errors
    embed the whole traced graph, so only the type and first line are
    kept).

Two health-rule fixes came out of running the demos:

-   **Update ratio per step.** `HealthMonitor` counts optimizer steps
    (`register_step_post_hook`) and divides the change since the previous
    `collect()` by that count. Collecting once per epoch had reported
    whole-epoch changes (≈0.1) against a per-step threshold (0.01),
    raising false "update ratio high" alerts in every demo.
-   **Idle GPU only for GPU scripts.** `gpu_underused` now requires the
    script to report a CUDA device; CPU-only scripts on a machine with a
    GPU no longer get "GPU 2 % busy" alerts.

## Consequences

-   One call links any exportable PyTorch model; demos for an MLP, a CNN,
    a self-attention model and an autoencoder exercise it end to end.
-   Some models cannot be exported (`nn.MultiheadAttention`'s fused
    inference kernel, custom ops); they stream without a graph. The
    attention demo writes attention out explicitly for this reason.
-   Name matching depends on PyTorch's scope naming; renamed or
    functional layers (`F.relu`) are not placed, and the page says so.
-   Exports count against the upload limit; very large models need
    `NEUROVISION_MAX_UPLOAD_BYTES` raised.
