# ADR 0003: Training jobs, event streaming, and real values

Status: accepted (Milestone 4)

## Context

Educational networks must train while the UI shows real loss,
accuracy, activations, weights, and gradients. Training must not block
the API, must be cancellable, and must not stream the full model state.

## Decision

-   **Explicit instrumentation, not hooks.** `EducationalNetwork.trace()`
    records each layer's `pre` (W·x + b) and `post` (after activation)
    as the forward pass computes them. Training uses `forward()`
    (logits only). Every value the UI shows comes from these calls.
-   **Seeded and isolated.** Weight initialization uses the config's
    `seed` inside `torch.random.fork_rng`. Datasets and batch order use
    their own `torch.Generator`. The same settings reproduce the same
    run exactly; tests assert this.
-   **Thread pool, per-model lock.** Jobs run in a `ThreadPoolExecutor`
    (default 2 workers; more jobs are rejected with `429` instead of
    queued). Each optimizer step, validation pass, forward request, and
    tensor read holds the model's lock, so readers never see half a step.
-   **Cancellation** is a `threading.Event`, checked every batch and
    waited on during epoch delays. Evicting a model and shutting down
    the app both cancel jobs.
-   **Events.** Each job keeps its own event history (at most
    epochs + 2 small events) and delivers to WebSocket subscribers with
    `loop.call_soon_threadsafe`. Subscribing snapshots the history and
    registers under one lock, so no event is lost or duplicated between
    replay and live delivery. Step events carry four metrics and at most
    8 × 256 activation values for one probe example, never weights.
-   **Divergence** (non-finite loss) stops the job with an actionable
    message instead of streaming NaNs.
-   **Frontend.** The stream hook validates every event with Zod,
    batches store updates (≤ 20/s), reconnects up to 3 times (replay
    makes this lossless; epochs are de-duplicated), and bumps a
    `weightsVersion` when a job ends. Tensor views and forward results
    are keyed by that version, so stale values are hidden, never shown
    as current.

## Consequences

-   Jobs and models are lost on restart (in-memory, per process).
    Multi-process deployment needs a shared job store and a separate
    worker (Milestone 9).
-   Gradients are a snapshot from the last batch of each epoch, not an
    average over the epoch.
-   Weight-colored connections and 3D activation coloring are not
    implemented yet. Real weights and activations are available
    through the inspector and the 2D unit shading.
