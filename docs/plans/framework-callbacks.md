# Plan: framework callbacks and pip packaging for the SDK

Goal: people who train with **PyTorch Lightning**, the **Hugging Face
`Trainer`** or **Keras** can stream to Live Runs by adding one callback,
without rewriting their loop; and the SDK installs with `pip`.

Done when:

-   each framework trains a small real model in a test, and the backend
    receives its metrics (and health, where supported) and a final status;
-   `pip install ./sdk` works in a fresh venv, offline;
-   every backend and frontend check passes (Phase 6);
-   the docs show a copy-paste example per framework.

---

## Phase 0: Starting state (already done)

-   [x] Installed into `backend/.venv`: `lightning` 2.6.6,
    `transformers` 5.20.2, `keras` 3.15.1. Keras runs on your PyTorch:
    set `KERAS_BACKEND=torch` **before** `import keras`, or it looks
    for TensorFlow.
-   [x] `sdk/neurovision_client.py` (uncommitted edits):
    -   `__version__ = "0.1.0"`;
    -   a `_forward(model, batch)` helper. `ActivationProbe` and
        `HealthMonitor` now call `model(**batch)` when the probe batch
        is a dict, which Hugging Face models need.
-   [x] Backend suite still passes: `304 passed`.
-   Note: once, right after installing, `import torch` failed with
    *"An Application Control policy has blocked this file … shm.dll"*.
    Retrying worked. If it happens again, wait a minute and rerun.

---

## Phase 1: Shared core (`sdk/neurovision_callbacks.py`)

Create **one new file** next to `neurovision_client.py`. It imports the
client and must **not** import any framework at module level.

### 1.1 Lazy class creation

Each callback must subclass its framework's base class. Lightning
checks `isinstance(cb, Callback)`, and so do Hugging Face and Keras. So
build each class only when the user asks for it:

``` python
def __getattr__(name: str):
    factories = {
        "LightningCallback": _make_lightning,
        "HuggingFaceCallback": _make_huggingface,
        "KerasCallback": _make_keras,
    }
    if name in factories:
        cls = factories[name]()
        globals()[name] = cls   # build once
        return cls
    raise AttributeError(name)
```

Then `from neurovision_callbacks import KerasCallback` works without
Lightning or Transformers installed.

### 1.2 `_Session`: the shared logic all three callbacks use

| Method | What it does |
|---|---|
| `start(name, total_epochs, model, optimizer, example_input, on_checkpoint, allow_set_lr)` | calls `nv.start_run(...)` with `control=...`; passes `optimizer` only if `allow_set_lr` |
| `record(step, epoch, metrics, progress=None, samples=None)` | merges into a **pending** entry for that step (see 1.3) |
| `flush()` | sends the pending entry with `run.log(...)`, attaching the latest health and activations |
| `poll() -> bool` | `run.poll()`; False means stop was requested |
| `finish(error=None)` | flush, then `run.finish()`; `"failed"` with the exception **type name only**; `"cancelled"` for `KeyboardInterrupt` |

Constructor options (same for all three callbacks):

| Option | Default | Meaning |
|---|---|---|
| `name` | model class name | run name (1–80 chars) |
| `url` | `$NEUROVISION_URL`, then `http://127.0.0.1:8000` | backend |
| `total_epochs` | from the framework | override when the framework doesn't know |
| `control` | `False` | allow pause/resume/stop (+ checkpoint, set_lr where possible) |
| `checkpoint_dir` | `None` | enables the checkpoint command |
| `health` | `True` | `HealthMonitor` (Lightning, Hugging Face only) |
| `activation_layers` | `None` | layer/module names for `ActivationProbe` |
| `probe_batch` | `None` | fixed batch for activations and health activation stats |
| `log_every_n_steps` | `None` | `None` = once per epoch (Lightning, Keras); Hugging Face follows `logging_steps` |
| `run_options` | `{}` | passed to `start_run` (e.g. `track_system`, `transport` in tests) |

### 1.3 Merge logs that share a step

The server rejects a step number that doesn't increase. Frameworks
often report twice at one step: Lightning batch-end then epoch-end,
Hugging Face train log then eval log, Keras last batch then epoch end.

Rule: keep one **pending** entry `{step, epoch, progress, metrics}`.

-   Same step comes in: merge the metrics.
-   Higher step comes in: `flush()` the old one, start a new one.
-   Each callback also calls `flush()` at points where it knows nothing
    more arrives for that step (listed per framework below). This keeps
    the delay to at most one logging interval.

### 1.4 Metric-name cleanup (`_normalize(metrics, framework)`)

So the Live Runs charts pair training with validation:

| Input | Output | Why |
|---|---|---|
| `val/loss` | `val_loss` | the UI pairs on `val_` |
| `train_loss`, `train/loss` | `loss` | pairs with `val_loss` (skip if `loss` already exists) |
| `loss_epoch` (Lightning) | `loss` | and drop `loss_step` when `loss_epoch` exists |
| `eval_loss`, `eval_accuracy` (Hugging Face) | `val_loss`, `val_accuracy` | |
| `eval_runtime`, `eval_samples_per_second`, `eval_steps_per_second`, `total_flos`, `train_runtime`, `epoch` | dropped | not training metrics |

Also: tensors become `float(t.item())` only when `numel() == 1`; skip
non-numbers. The client already caps metrics at 16.

### 1.5 Checkpoint file names

Write into `checkpoint_dir`. Return only the **file name** (never the
full path) from `on_checkpoint`, because that text is sent to the
server. Example: `neurovision-step1200.ckpt`.

---

## Phase 2: The three callbacks

### 2.1 `LightningCallback` (`lightning.pytorch.Callback`)

Try `import lightning.pytorch as pl`, then fall back to
`import pytorch_lightning as pl`.

| Lightning hook | Do |
|---|---|
| `on_train_start(trainer, pl_module)` | only if `trainer.is_global_zero`. Work out `total_epochs`: `trainer.max_epochs` if > 0, else `ceil(trainer.max_steps / trainer.num_training_batches)`, else 1 with a warning. `optimizer = trainer.optimizers[0]`. Allow set_lr only if `not trainer.lr_scheduler_configs`. Checkpoint: `trainer.save_checkpoint(path)`. Graph link: if `link_graph=True` and `pl_module.example_input_array` is set, pass `model=pl_module, example_input=...`. Create `HealthMonitor(pl_module, optimizer=..., probe_batch=...)` and `ActivationProbe.by_name(...)` if asked. |
| `on_train_batch_start(...)` | `if not session.poll(): trainer.should_stop = True` |
| `on_before_zero_grad(trainer, pl_module, optimizer)` | runs after `optimizer.step()` with gradients still present: collect health **only on logging steps** |
| `on_train_batch_end(...)` | if `log_every_n_steps` divides `trainer.global_step`: `record(step=global_step, epoch=current_epoch+1, progress=current_epoch + (batch_idx+1)/num_training_batches, metrics=callback_metrics)` |
| `on_train_epoch_end(...)` | validation has already run, so `callback_metrics` holds train **and** val. `record(...)`, then `flush()` |
| `on_exception(trainer, pl_module, exception)` | `finish(error=exception)` |
| `on_fit_end(...)` | `finish()` |

Ignore everything while `trainer.sanity_checking` is true.

### 2.2 `HuggingFaceCallback` (`transformers.TrainerCallback`)

Verified in `transformers/trainer.py` 5.20.2: `on_pre_optimizer_step`,
then `optimizer.step()`, then **`on_optimizer_step`**, then
`lr_scheduler.step()`, then `zero_grad()`, then `global_step += 1`, then
`on_step_end`, then logging and eval (which call `on_log`). Every hook
also receives `model=`, `optimizer=`, `lr_scheduler=` in `**kwargs`.

| Hugging Face hook | Do |
|---|---|
| `on_train_begin(args, state, control, model, optimizer, **kw)` | only if `state.is_world_process_zero`. `total_epochs = ceil(state.num_train_epochs or args.num_train_epochs)`. **No set_lr**: the scheduler overwrites the rate every step. Checkpoint: `model.save_pretrained(dir/neurovision-step{N})` (weights only; say so in the reply). `HealthMonitor(model, optimizer=optimizer)`. |
| `on_step_begin` | `if not session.poll(): control.should_training_stop = True` |
| `on_optimizer_step` | collect health if `(state.global_step + 1)` is a logging step |
| `on_log(args, state, control, logs)` | skip the final summary (has `train_runtime`). `record(step=state.global_step, epoch=max(1, ceil(state.epoch)), progress=state.epoch, metrics=logs, samples=steps_since_last * args.train_batch_size * args.gradient_accumulation_steps)` |
| `on_step_end` / `on_epoch_end` | `flush()` if pending is for an earlier step / always |
| `on_train_end` | `finish()` |

Activations: `probe_batch` can be a tokenizer output dict, e.g.
`tokenizer("a test", return_tensors="pt")`; the `_forward` helper
handles it. Graph link: **skip** (dict inputs don't export cleanly);
document that.

### 2.3 `KerasCallback` (`keras.callbacks.Callback`)

| Keras hook | Do |
|---|---|
| `on_train_begin(logs)` | `total_epochs = self.params["epochs"]`. set_lr: wrap the optimizer in an adapter (below), but only if `self.model.optimizer._learning_rate` is **not** a `keras.optimizers.schedules.LearningRateSchedule`. Checkpoint: `self.model.save(dir/neurovision-step{N}.keras)`. |
| `on_train_batch_begin` | `if not session.poll(): self.model.stop_training = True` |
| `on_train_batch_end(batch, logs)` | if `log_every_n_steps`: `step = epoch * self.params["steps"] + batch + 1` |
| `on_epoch_end(epoch, logs)` | `logs` already has `val_*`. `record(step=(epoch+1)*steps, epoch=epoch+1, ...)`, then `flush()` |
| `on_train_end` | `finish()` |

Learning-rate adapter (the client's `set_lr` writes
`optimizer.param_groups[*]["lr"]`):

``` python
class _KerasLr:
    def __init__(self, opt): self.opt = opt
    def __getitem__(self, key):            # key == "lr"
        return float(keras.ops.convert_to_numpy(self.opt.learning_rate))
    def __setitem__(self, key, value):
        self.opt.learning_rate = value     # Keras setter

class _KerasOptimizer:
    def __init__(self, opt): self.param_groups = [_KerasLr(opt)]
```

Activations: build a sub-model
`keras.Model(model.inputs, [model.get_layer(n).output for n in names])`,
call it on `probe_batch`, keep the first example, flatten, first 256
values. This needs a functional or Sequential model; for subclassed
models, warn and skip. No `HealthMonitor` for Keras (it reads PyTorch
`.grad`).

---

## Phase 3: pip packaging

### 3.1 `sdk/pyproject.toml`

``` toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "neurovision-client"
version = "0.1.0"            # keep equal to neurovision_client.__version__
description = "Stream your own training to NeuroVision Live Runs"
readme = "README.md"
requires-python = ">=3.10"
dependencies = []            # standard library only

[project.optional-dependencies]
torch = ["torch>=2.1"]
system = ["psutil", "nvidia-ml-py"]
lightning = ["lightning>=2.0"]
transformers = ["transformers>=4.40", "torch>=2.1"]
keras = ["keras>=3.0"]

[tool.setuptools]
py-modules = ["neurovision_client", "neurovision_callbacks"]
```

No `license` field: the repository has no license yet (README says
"all rights reserved"). Add one when you choose a license.

### 3.2 `sdk/README.md`

The package page: install, a three-line example per framework, and a
link to `docs/live-runs.md`.

### 3.3 Install forms to document

``` bash
pip install ./sdk
```
``` bash
pip install "neurovision-client[lightning] @ git+https://github.com/vishwa-10147/neurovision#subdirectory=sdk"
```

Copying the single file still works for plain loops. Callbacks need
both files.

---

## Phase 4: Tests (`backend/tests/test_callbacks.py`)

Reuse the `live_server` fixture idea from `tests/test_client.py`: a real
uvicorn on a free port. Read results back with
`GET /api/v1/runs/{id}` and the event WebSocket.

At the top of the file, before importing keras:
`os.environ.setdefault("KERAS_BACKEND", "torch")`.

| Test | Checks |
|---|---|
| core: `_normalize` table | every row of the 1.4 table |
| core: step merging | same step merges; higher step flushes; nothing lost at finish |
| core: framework not installed | `import neurovision_callbacks` works with `sys.modules["lightning"] = None` |
| Lightning end to end | tiny `LightningModule` (2→8→2), 3 epochs, `limit_train_batches=4`, logs `train_loss`/`val_loss`. Run is `completed`; steps 1 per epoch, increasing; metrics include `loss` and `val_loss`; a step has `health` |
| Lightning stop | control on; issue `POST /runs/{id}/commands {"kind":"stop"}` after the first epoch; run ends `cancelled` before `max_epochs` |
| Lightning crash | `training_step` raises `ValueError` at batch 2; run ends `failed` with "…raised ValueError." |
| Hugging Face end to end | tiny model built from a config, **no download**: `BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)` and `BertForSequenceClassification`, a 32-row dataset, `logging_steps=2`, `eval_strategy="epoch"`, `report_to=[]`. Metrics include `val_loss`; no `eval_runtime`; `set_lr` **not** in controls |
| Keras end to end | Sequential `Dense(8, relu)` → `Dense(2)`, `validation_split=0.25`, 3 epochs. Metrics include `val_loss`; `set_lr` applies and changes `optimizer.learning_rate` |
| Keras + schedule | with `ExponentialDecay`, `set_lr` is **not** offered |
| packaging | Phase 6, step 4 (manual) |

Keep each test under ~10 s (tiny data, CPU).

Add to `backend/requirements-dev.txt` the exact installed versions:

```
lightning==2.6.6
transformers==5.20.2
keras==3.15.1
```

---

## Phase 5: Docs and examples

-   [ ] `sdk/examples/train_lightning.py`, `train_hf_trainer.py`,
    `train_keras.py`: small, runnable, no downloads, `--url` argument
    like `train_spiral.py`.
-   [ ] `sdk/examples/README.md`: add the three rows to the demo table
    (fill in results only from real runs).
-   [ ] `docs/live-runs.md`: a section "Using a training framework" with
    a snippet per framework, the controls each one supports, and the
    known limits (Phase 7).
-   [ ] `README.md`: the "Watch your own PyTorch training live" section
    gets one line per framework plus `pip install ./sdk`; add a
    roadmap row.
-   [ ] `docs/decisions/0011-framework-callbacks.md`: an ADR covering
    context, decision (lazy classes, step merging, name cleanup, which
    controls per framework and why), alternatives (one generic logging
    handler; a TensorBoard bridge), consequences.

---

## Phase 6: Verify

Run from `backend/` with the venv:

1.  `.venv/Scripts/python -m pytest -q`: everything passes, including
    the new tests.
2.  `.venv/Scripts/ruff check . ../sdk` and
    `.venv/Scripts/ruff format --check . ../sdk`
3.  Frontend (nothing should change, but confirm), from `frontend/`:
    `npm run lint`, `npm run typecheck`, `npm test`, `npm run build`
4.  Packaging, offline:
    ```
    .venv/Scripts/python -m pip wheel --no-build-isolation --no-deps ../sdk -w <scratch>/dist
    python -m venv <scratch>/clean
    <scratch>/clean/Scripts/python -m pip install --no-index <scratch>/dist/neurovision_client-0.1.0-py3-none-any.whl
    <scratch>/clean/Scripts/python -c "import neurovision_client, neurovision_callbacks; print(neurovision_client.__version__)"
    ```
    The last line must print `0.1.0` without any framework installed.
5.  Live check: start the `backend` and `frontend` servers
    (`.claude/launch.json`, ports 8001/5174), run each example with
    `--url http://127.0.0.1:8001`, and watch it in **Live runs**:
    charts pair `loss`/`val_loss`, the run ends `Completed`, and pause
    and stop work where offered.

---

## Phase 7: Known limits to state in the docs

-   Hugging Face and Keras have no exception hook. A crash shows as
    *failed* only when the script exits (the client's `atexit`
    handler), not at the moment of the crash.
-   Hugging Face checkpoints from the dashboard save model weights
    only, not the optimizer or scheduler state.
-   No `set_lr` for Hugging Face (the scheduler overwrites it), for
    Lightning with a scheduler, or for Keras with a schedule.
-   Keras: no health statistics. Activations need a functional or
    Sequential model.
-   Distributed training: only rank 0 / the main process logs.

---

## When you update me

Tell me which phase you finished and paste any failing command output.
I can review the diff, run the checks, or take over a phase.
