# ADR 0011: Framework callbacks and pip packaging for Live Runs

Status: accepted

## Context

Live Runs previously required a hand-written loop calling
`neurovision_client.start_run()` and `run.log()`. That works for the raw
PyTorch examples, but most real projects use a trainer framework. Requiring
users to rewrite those loops made the feature unavailable to PyTorch
Lightning, Hugging Face Trainer and Keras projects. The SDK also needed a
normal install path without forcing every framework on every user.

## Decision

1. Add `sdk/neurovision_callbacks.py` with three lazy-created callback
   classes. The module imports only the standard library and
   `neurovision_client` at module import time. `LightningCallback`,
   `HuggingFaceCallback` and `KerasCallback` are built by `__getattr__` only
   when requested, and each subclasses its framework's real callback base.
2. Put lifecycle, polling, checkpoint naming, health/activation capture and
   failure handling in one private `_Session`. This keeps the three adapters
   consistent while preserving framework-specific hook behavior.
3. Merge records that share a framework step before calling the backend. The
   backend requires strictly increasing step numbers, while all three
   frameworks can emit a batch, epoch, evaluation or logging record for the
   same step.
4. Normalize metric names into the chart vocabulary: `eval_loss` and
   `val/loss` become `val_loss`; training-loss variants become `loss`; and
   runtime/throughput summary fields are omitted. Only finite scalar metrics
   are forwarded.
5. Expose controls only where the framework can apply them safely. Lightning
   may set a scalar optimizer learning rate when no scheduler is configured;
   Hugging Face never exposes `set_lr` because its scheduler owns the value;
   Keras uses a small adapter for scalar learning rates and disables the
   control for schedules. Checkpoints return only a filename in command
   acknowledgements.
6. Package both SDK modules with `sdk/pyproject.toml`. The base package has no
   required dependencies. Framework and system integrations are optional
   extras, and copying the two SDK files remains supported.

## Alternatives considered

- **One generic logging handler:** trainer hooks differ in ordering, control
  semantics and failure behavior. A generic handler would lose validation and
  create duplicate or rejected steps.
- **TensorBoard bridge:** it would add another storage/translation path,
  cannot provide NeuroVision's pull-based controls, and would not naturally
  carry selected activations or health measurements.
- **Import every framework at module import time:** this makes a lightweight
  client unusable in environments that use only one framework and breaks the
  standard-library-only install promise.

## Consequences

- Users can add one callback to each supported trainer and keep their existing
  training code.
- The SDK remains importable without Lightning, Transformers or Keras.
- Hugging Face and Keras cannot report a crash until the process exits because
  their callback APIs do not provide a dependable exception hook.
- Hugging Face checkpoints contain model weights only; optimizer and scheduler
  state are not saved.
- Keras has no PyTorch-style `HealthMonitor` gradient statistics, and its
  activation probe requires Functional or Sequential model inputs.
- Distributed runs intentionally log only rank zero / world-process zero.
