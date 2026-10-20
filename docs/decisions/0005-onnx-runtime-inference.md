# ADR 0005: Running imported models in an isolated ONNX Runtime worker

Status: accepted (Milestone 6)

## Context

Imported ONNX models are untrusted. Users want to run them on inputs and
see real outputs and intermediate values. A malformed or hostile model
can crash, hang, or exhaust memory inside the runtime's native code.

## Decision

-   **Process isolation.** The API process never imports ONNX Runtime; a
    test asserts this. One long-lived worker process
    (`app/ml/onnx_worker.py`, started with `spawn`) holds up to four
    cached sessions. The parent (`app/services/onnx_runtime.py`) talks
    to it over a pipe.
-   **Limits.**
    -   Every run has a wall-clock limit (`NEUROVISION_INFERENCE_TIMEOUT_SECONDS`,
        default 10 s). On timeout the worker is killed, the request gets
        `504 inference_timeout`, and a fresh worker starts on the next run.
    -   Waits check every 50 ms whether the worker is still alive, so a
        crash returns `500 inference_crashed` at once.
    -   On Linux and macOS the worker's address space is capped
        (`NEUROVISION_INFERENCE_MAX_MEMORY_BYTES`, default 2 GiB) with
        `RLIMIT_AS`. Windows has no equivalent rlimit, so there the
        process boundary and the time limit are the protections. Deploy
        on Linux, or in a memory-limited container, if you run untrusted
        models.
    -   Only the CPU execution provider is used, so no provider that could
        reach the network is ever enabled.
    -   One run at a time; a second caller waits briefly, then gets `429`.
-   **Validated inputs.** Inputs must match the model's declared input
    names and ranks. Fixed dimensions must match exactly, named or
    unknown dimensions take any size from 1 to 100,000, and the total is
    at most 1,000,000 values. The dtype must be supported. Integer values
    must be whole and in range. Inputs can be typed, or generated
    deterministically from a seed (`random`, `zeros`, `ones`, `range`).
-   **Capability from the file.** At import, a model is marked runnable
    unless it uses operators outside the ONNX standard (except ONNX
    Runtime's own `com.microsoft` contrib ops), keeps weights in
    external files, has unsupported input types, or the runtime is not
    installed. The reasons are returned as `inference_blockers`.
-   **Intermediate values.** For graphs up to 200 operators, a variant of
    the model is built in the parent (no execution) that exposes every
    operator output as a graph output. Larger graphs capture only the
    operators named in `capture`. Basic graph optimization keeps those
    tensors intact.
-   **Bounded results.** The worker summarizes each tensor (shape,
    finite min/max/mean/std, NaN/∞ count, at most 4,096 values), so huge
    tensors never cross the process boundary or reach the browser.
-   **Honest timing.** `elapsed_ms` is model execution measured inside
    the worker. Model loading and worker startup are reported as notes.

## Consequences

-   Verified with a real PyTorch 2.14.1 export: outputs matched PyTorch's
    own logits to 4 decimals, and a test checks a PyTorch export to
    1e-5.
-   A single worker serializes runs; a pool would be needed for many
    concurrent users (Milestone 9).
-   Values are shown per operator in 2D (block color: output RMS) and in
    the inspector; per-channel feature-map views are Milestone 7 work.
