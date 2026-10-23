# Watching your own training live

The **Live runs** view shows a training script from your own project
while it trains: its loss, accuracy (or any metrics you choose), and the
activations of layers you pick, updating as the script logs them. The
script keeps running on your machine with your data and hardware;
NeuroVision only receives the small values you send.

## Set up

1.  Start NeuroVision (see the README). Note the backend URL, e.g.
    `http://127.0.0.1:8000`.
2.  Copy [`sdk/neurovision_client.py`](../sdk/neurovision_client.py)
    next to your training script. It is one file and needs only the
    Python standard library (plus PyTorch for `ActivationProbe`).
3.  Open **Live runs** in the browser and start your script. The run
    appears within a few seconds; click it to watch.

Use `127.0.0.1` rather than `localhost` in the URL. On Windows,
`localhost` tries IPv6 first and costs about 2 seconds per request when
the backend listens on IPv4 only (measured: 2.07 s vs 0.002 s).

## Add it to a standard PyTorch loop

``` python
import neurovision_client as nv

EPOCHS = 20
run = nv.start_run("resnet18 / lr 3e-4", total_epochs=EPOCHS,
                   url="http://127.0.0.1:8000")   # or set NEUROVISION_URL

# Optional: post-activation values of chosen layers for one fixed example.
probe = nv.ActivationProbe(
    model,
    {"layer1 ReLU": model.layer1[0].relu, "avgpool": model.avgpool},
    probe_batch=next(iter(val_loader))[0][:1].to(device),
)

with run:                                   # marks the run failed on an exception
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        run.log(
            epoch=epoch,
            metrics={"loss": train_loss, "accuracy": train_acc,
                     "val_loss": val_loss, "val_accuracy": val_acc},
            activations=probe.capture(),
        )
# Leaving the `with` block calls run.finish().
```

Without `with`, call `run.finish()` at the end (or
`run.finish("failed", "why")`).

A complete, runnable example trains a small MLP on a synthetic spiral
(no downloads):

``` bash
python sdk/examples/train_spiral.py --url http://127.0.0.1:8001
```

### What to log

-   **`metrics`**: up to 16 named numbers per step. Floats, NumPy
    scalars, and 0-d tensors all work. `val_x` (or `x_val`, `test_x`)
    is charted with `x`; accuracy-like metrics in `[0, 1]` are shown as
    percentages.
-   **`step`**: optional. Each logged step needs a larger step number
    than the last; by default the client counts 0, 1, 2, … Pass your
    global batch count to make the x-axis match your own logs.
-   **`activations`**: up to 8 layers, each up to 256 values (longer
    outputs keep their first 256). `ActivationProbe` produces this
    dictionary: it attaches forward hooks only while `capture()` runs,
    feeds the probe batch in `eval()` mode under `torch.no_grad()`,
    keeps the first example, and restores the model's previous mode.
    To see post-activation values, probe the activation modules
    (`nn.ReLU`, `nn.GELU`, …) or blocks that end in one.

A run accepts at most 10,000 logged steps. Logging every batch of a long
run reaches that quickly, so log every N batches or once per epoch.

## Training health and alerts

`HealthMonitor` measures the signals that usually show a run going wrong,
and the backend turns them into alerts in the **Health alerts** panel:

``` python
monitor = nv.HealthMonitor(
    model,
    optimizer=optimizer,          # reports the learning rate
    probe_batch=x_val[:64],       # optional: activation statistics
)
...
loss.backward()
optimizer.step()
run.log(epoch=epoch, metrics={"loss": loss, "val_loss": val_loss},
        health=monitor.collect())   # right after optimizer.step()
```

`collect()` measures, per module that owns parameters, the **gradient
norm**, **weight norm** and **update ratio** ‖Δw‖/‖w‖ since the previous
`collect()` (around 1e-3 is a common target), plus the total gradient
norm and the number of NaN/∞ gradients. With a `probe_batch`, it also
measures each `nn.ReLU`/`ReLU6`/`LeakyReLU`/`ELU`/`GELU`/`SiLU`/`Tanh`/
`Sigmoid` module's output: mean, std, the fraction of **dead units**
(≤ 0 for every probe example, ReLU family) and of **saturated outputs**
(tanh, sigmoid).

-   Call it right after `optimizer.step()`, before the next
    `zero_grad()`, as often as you log (it costs one pass over the
    parameters, plus one forward pass on the probe batch).
-   Update ratios keep one copy of the parameters on their device; pass
    `track_updates=False` to skip them for very large models.
-   Functional activations (`F.relu`) are not seen; a module used more
    than once reports its last use. Models with more than 32 parameter
    groups are grouped by top-level module.
-   Even without a monitor, a metric that becomes NaN or infinite is
    reported by name, so the dashboard raises a critical alert instead of
    the value silently disappearing.

Alerts fire once when a problem starts and again only after it has
cleared: non-finite loss or gradients (critical), exploding or vanishing
gradients, update ratios far too large (> 0.1) or small, dead or
saturated units, loss spikes, plateaus, and overfitting (`val_loss`
rising while `loss` falls). Each one quotes the measured numbers and
suggests what to try. The rules are listed in
[ADR 0008](decisions/0008-run-health-and-controls.md).

## Controlling the run from the dashboard

Opt in when starting the run, then call `run.poll()` in your loop:

``` python
def save():
    torch.save(model.state_dict(), "checkpoint.pt")
    return "checkpoint.pt"            # shown in the dashboard

with nv.start_run("resnet18", total_epochs=EPOCHS, control=True,
                  optimizer=optimizer, on_checkpoint=save) as run:
    for epoch in range(1, EPOCHS + 1):
        for xb, yb in train_loader:
            if not run.poll():        # applies commands; blocks while paused
                break                 # stop was requested
            ...                       # train one batch
        if run.should_stop:
            break
        run.log(...)
# A stopped run finishes as "cancelled" (shown as Stopped).
```

| Button | Needs | What the script does |
|--------|-------|----------------------|
| Pause / Resume | `control=True` | `run.poll()` blocks until resumed (still answering the other commands) |
| Stop | `control=True` | `run.poll()` returns `False` from then on; save and leave your loop |
| Save checkpoint | `on_checkpoint=` | calls your function with no arguments in the training thread |
| Learning rate → Set | `optimizer=` | sets `lr` on every parameter group (a scheduler that sets `lr` itself will override it) |

-   Commands are **applied in your training thread**, at the `poll()`
    you choose, never from a background thread in the middle of a step.
    `poll()` only checks an in-memory queue, so call it once per batch.
-   The client fetches commands about once a second (and with every
    event batch); nothing connects *into* your machine.
-   The dashboard shows each command as *Waiting for the script…* until
    the script answers *Applied* or *Rejected* with a reason (for
    example "The run is not paused." or "on_checkpoint raised OSError.").
    Commands still pending when the run ends are marked *Expired*.

**Security:** anyone who can reach the backend can send commands to a
run that opted in, just as anyone can watch runs. Controls are therefore
off by default; enable them only with the backend on localhost or a
trusted network.

## Effect on training speed

`run.log()` converts the values, appends them to a bounded in-memory
queue, and returns; a background thread sends batches of up to 32
steps every 0.5 s. When the backend is slow or unreachable, the oldest
queued steps are dropped (counted in `run.dropped`) instead of blocking
your loop, and the client logs one warning. Nothing in the client
raises into your training code:

-   Backend unreachable at start: a warning, then a run that ignores
    every call.
-   Values the server would reject (more than 16 metrics, non-finite
    activations, …): dropped with a one-time warning. A NaN or infinite
    metric is not sent, but its name is reported so the dashboard alerts.
-   The run is finished, full, or rejected by the server: logging stops
    with a warning.

The one cost you control is `ActivationProbe.capture()`: one extra
forward pass on the probe batch (use a single example) each time you
call it.

## Security

`start_run` returns a write token that only the creating script holds;
the backend keeps only its SHA-256 digest. Logging and finishing need
`Authorization: Bearer <token>`, so other clients can list and watch
runs but cannot add events to them. Error messages sent when a `with`
block fails contain only the exception type (e.g. `The training script
raised RuntimeError.`), never its message, which can contain paths or
data.

The backend has no user accounts yet: anyone who can reach it can
create runs and watch them. Keep it on a trusted network.

## Limits and behavior

| Rule                       | Value                                           |
|----------------------------|-------------------------------------------------|
| Active runs                | 8 (`NEUROVISION_MAX_EXTERNAL_RUNS`); more → `429` |
| Steps per run              | 10,000; more → `409`                            |
| Steps per request          | 32; request body at most 2 MB                   |
| Metrics / layers / values  | 16 / 8 / 256 per step, finite numbers only      |
| Idle timeout               | 5 minutes without steps or heartbeats → run marked failed |
| Retention                  | the 32 most recent finished runs, in memory     |
| Health per step            | 32 parameter groups, 32 activation layers       |
| Alerts per run             | 200, then one "suppressed" notice               |
| Pending commands           | 16 per run; learning rate in (0, 10]            |

The client sends an empty batch (a heartbeat) after 60 seconds without
steps, so long epochs do not time out. If the script crashes without
finishing, the run is closed after the idle timeout with an explanation.
Runs live in the backend's memory and are lost when it restarts.

The REST endpoints are described in [api.md](api.md#external-runs) and
the design in [ADR 0006](decisions/0006-external-runs.md) and, for health
and controls, [ADR 0008](decisions/0008-run-health-and-controls.md). The
example [`sdk/examples/train_spiral.py`](../sdk/examples/train_spiral.py)
uses all of it; run it with `--lr 0.5` to see alerts fire.
