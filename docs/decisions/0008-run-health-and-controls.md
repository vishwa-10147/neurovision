# ADR 0008: Training health monitoring and two-way run controls

Status: accepted

## Context

Live runs (ADR 0006) show a user's own training while it happens, but only
as a one-way stream of metrics and activations. Two things are missing to be
useful during real training:

1.  **Health.** Most failed runs announce themselves early: gradients
    explode or vanish, ReLUs die, tanh/sigmoid units saturate, updates are
    far too large or too small for the weights, the loss turns NaN, spikes,
    plateaus, or validation loss rises while training loss falls. Users
    should see these signals, and be told about them, without staring at
    every chart.
2.  **Control.** When a run goes wrong, users want to act from the
    dashboard: pause to inspect, change the learning rate, save a
    checkpoint, or stop, without killing the process and losing state.

NeuroVision must still never run or import user code, never need an
inbound connection to the user's machine, and keep every payload bounded.

## Decision

### Health statistics are measured in the script

-   `HealthMonitor` in `sdk/neurovision_client.py` measures, when the
    script asks (`monitor.collect()`, right after `optimizer.step()`):
    -   per parameter group (a module that owns parameters): gradient L2
        norm, weight L2 norm, and **update ratio** ‖Δw‖ / ‖w‖ since the
        previous `collect()` (needs one copy of the parameters; can be
        turned off with `track_updates=False`);
    -   globally: total gradient norm, the optimizer's learning rate, and
        the number of non-finite gradient values;
    -   on a fixed probe batch, for activation modules (`ReLU`, `ReLU6`,
        `LeakyReLU`, `ELU`, `GELU`, `SiLU`, `Tanh`, `Sigmoid`): mean and
        standard deviation, the **dead fraction** (units, i.e. channels for
        conv outputs, that are ≤ 0 for every probe example; ReLU family
        only), and the **saturated fraction** (|y| > 0.99 for tanh, y < 0.01
        or y > 0.99 for sigmoid).
    Activations use forward hooks attached only during `collect()`, in eval
    mode under `no_grad`, like `ActivationProbe`. Functional activations
    (`F.relu`) are not seen; this is stated in the docs.
-   The client always reports metrics that were non-finite (NaN/∞) by
    name in `health.nonfinite_metrics`, even without a monitor, instead of
    silently dropping them as before.
-   Bounds: at most 32 parameter groups and 32 activation layers per step
    (groups are merged by top-level module, then into `other`, when a model
    has more), finite non-negative numbers or `null`.

### Alerts are decided on the server

`app/services/run_health.py` is a pure, unit-tested rule engine fed every
logged step. It turns statistics into `run.alert` events with a severity,
the layer (if any), a message with the measured numbers, and advice:

| Alert | Rule | Severity |
|-------|------|----------|
| `non_finite_loss` | a metric was NaN/∞ | critical |
| `non_finite_gradients` | any non-finite gradient value | critical |
| `exploding_gradients` | gradient norm > 10 × median of the last 20 (≥ 5 seen) | warning |
| `vanishing_gradients` | a group's gradient norm < 1e-7 for 3 health steps | warning |
| `update_ratio_high` | ‖Δw‖/‖w‖ > 0.1 | warning |
| `update_ratio_low` | ‖Δw‖/‖w‖ < 1e-6 for 3 health steps | info |
| `dead_units` | dead fraction ≥ 50 % for 3 health steps | warning |
| `saturated_units` | saturated fraction ≥ 50 % for 3 health steps | warning |
| `loss_spike` | loss > 3 × median of the last 10 (≥ 5 seen) | warning |
| `loss_plateau` | no > 1 % improvement of the best loss for 20 logged steps | info |
| `overfitting` | `val_loss` rose 5 logged steps in a row while `loss` fell | warning |
| `gpu_underused` | a GPU's NVML utilization ≤ 20 % for 5 logged steps (per GPU; from `system`) | info |
| `gpu_memory_high` | a GPU's memory ≥ 95 % used (per GPU; from `system`) | warning |

An alert fires once when its condition starts and re-arms when it clears,
so a persistent problem is one alert, not one per step. A run publishes at
most 200 alerts. Thresholds are conventional heuristics (for example an
update ratio around 1e-3 is often cited as healthy); the UI presents
alerts as signals to check, not verdicts.

### Controls are opt-in, pulled by the script, and acknowledged

-   A run accepts commands only if the script opted in:
    `start_run(..., control=True)` enables `pause`, `resume` and `stop`;
    passing `optimizer=` adds `set_lr`; `on_checkpoint=` adds
    `checkpoint`. The run's `controls` list says which, and the UI shows
    only those buttons.
-   The browser issues a command with `POST /runs/{id}/commands`. The
    script **pulls** pending commands: they ride on the response of every
    event batch, and the client's background thread also polls
    `GET /runs/{id}/commands` (write token required) about once a second.
    No inbound connection to the user's machine is ever needed.
-   Commands are **applied in the training thread** at a point the script
    chooses: `run.poll()` (e.g. once per batch). `poll()` applies new
    commands, blocks while paused (still answering `resume`, `stop`,
    `set_lr` and `checkpoint`), and returns `False` once a stop was
    requested, so the script can save and exit cleanly. Changing the
    learning rate writes `lr` on every optimizer parameter group; a
    scheduler that sets `lr` itself will override it, which the docs state.
-   Every command is acknowledged by the script (`applied` or `rejected`
    with a reason) through `POST /runs/{id}/commands/{command_id}/ack`, and
    each state change is published as a `run.command` event, so the UI
    shows *pending → applied/rejected*, never an assumed outcome. Pending
    commands expire when the run ends.
-   A run stopped this way finishes with the new status `cancelled`
    (`training.cancelled` event), distinct from `completed` and `failed`.
-   Bounds: at most 16 pending commands per run; `set_lr` values must be
    finite, > 0 and ≤ 10; an applied pause/learning-rate change updates the
    run's `paused` / `learning_rate` fields.

### Bounded history

Step events keep their full health details only for the latest 200 steps
that carried health; older steps keep the global numbers (gradient norm,
learning rate, non-finite counts), so trend charts still work and memory
stays bounded.

## Alternatives considered

-   **Server-pushed commands over a WebSocket to the script.** Needs a
    WebSocket library and reconnect logic in the user's environment, and a
    long-lived connection; polling every second is simpler and fast enough
    for human-initiated actions.
-   **Computing alerts in the client.** Would duplicate the rules in every
    client language and make them untestable from the server; the server
    sees every step anyway.
-   **Automatic actions on alerts** (auto-pause on NaN). Useful, but acting
    on a heuristic without the user is surprising; left for later as an
    explicit opt-in.

## Consequences

-   Anyone who can reach the backend can issue commands to runs that opted
    in, as anyone can watch runs today; tokens protect only the script's
    writes. Controls are therefore off by default, and authentication
    remains future work. Run the backend on localhost or a trusted
    network when controls are enabled.
-   `run.poll()` must be called for commands to take effect; a script that
    never calls it simply never applies them (commands stay pending, then
    expire).
-   Update ratios cost one extra copy of the parameters on their device.
