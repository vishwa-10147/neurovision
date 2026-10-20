# ADR 0006: External runs from users' own training scripts

Status: accepted

## Context

Users want to watch real projects train, not only the built-in
educational networks. Their training runs in their own scripts, on their
own data and hardware, often on another machine or GPU. NeuroVision
already streams bounded training events over a replayable WebSocket
(ADR 0003) and renders them in the Training Lab.

## Decision

-   **Push, not pull.** The user's script pushes metrics and activations
    to `POST /api/v1/runs/{run_id}/events`; NeuroVision never runs or
    imports user code. A run is created with `POST /api/v1/runs`.
-   **Same event format and stream.** Each external run is an
    `EventStream` (extracted from `TrainingJob`), and the server stamps
    `schema_version`, `job_id`, and `event_type` on logged steps, so
    `WS /api/v1/training/{id}/events` follows runs and jobs alike, with
    the same replay-then-follow semantics. Logged steps use the
    `training.step` bounds: 16 metrics, 8 layers × 256 values, finite
    numbers only.
-   **Bounded memory.** A run accepts at most 10,000 steps, 32 per
    request, 2 MB per body (read with a cap before parsing). Only the
    latest step with activations keeps them in the replay history, since
    the UI shows only the latest. At most 8 runs are active; the 32 most
    recent finished runs are retained.
-   **Write tokens.** Creation returns a random token once
    (`secrets.token_urlsafe(32)`); the server stores its SHA-256 digest
    and compares with `hmac.compare_digest`. Listing and watching stay
    open, like the rest of the API, but only the creator can write.
-   **Strict ordering, all-or-nothing batches.** Step numbers must
    increase and epochs must not decrease; a batch is fully validated
    before any of it is published. This lets clients de-duplicate
    replayed events by step.
-   **Idle timeout instead of liveness tracking.** A run with no events
    or heartbeats for 5 minutes is marked failed with an explanation.
    Expiry is checked lazily on API access (the Live runs view polls the
    list every 3 s), so no background timer is needed.
-   **A dependency-free client.** `sdk/neurovision_client.py` uses only
    the standard library. `log()` enqueues into a bounded deque and
    returns; a daemon thread sends batches, drops the oldest steps under
    backpressure, and never raises into the training loop.
    `ActivationProbe` attaches forward hooks only during `capture()`, so
    training forward passes pay nothing.
-   **Generic metrics in the UI.** External runs choose their own metric
    names, so the Live runs view charts every metric by step, pairing
    `x` with `val_x`, instead of the Training Lab's fixed four metrics.
    It reuses the stream hook (`useEventStream`), the chart component,
    and the activation strips.

## Alternatives considered

-   **Reading TensorBoard or W&B logs.** Would add heavy dependencies
    and does not carry activations in a bounded, live form.
-   **A WebSocket from the script.** Needs a WebSocket library in the
    user's environment and reconnect logic there; batched HTTP POSTs
    every 0.5 s are simple and fast enough to watch.
-   **Binding activations to imported ONNX graph nodes.** PyTorch module
    names rarely match exported ONNX node names. Runs can carry an
    optional `model_id` link, but activations are keyed by the names the
    script chooses.

## Consequences

-   Runs are in memory, per process, and lost on restart, like jobs and
    models.
-   Anyone who can reach the backend can create and watch runs; tokens
    only protect writes. Authentication is still future work.
-   The x-axis is the script's step number, which is only as meaningful
    as what the script passes.
