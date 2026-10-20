# API reference

Base path: `/api/v1`. Every error response uses the shared envelope
described in [schemas.md](schemas.md#error-envelope). Interactive docs
are served at `/docs` while the backend is running.

## `GET /health`

`200 OK`. The body follows `shared/schemas/health.schema.json`:

``` json
{"status": "ok", "service": "neurovision-backend", "version": "0.1.0", "environment": "development"}
```

## `POST /models/educational`

Creates an educational dense network from a configuration and returns
its normalized graph. The request body follows
`shared/schemas/dense-network-config.schema.json`:

``` json
{
  "name": "Two-moons classifier",
  "input_units": 2,
  "hidden_layers": [
    { "units": 8, "activation": "relu" },
    { "units": 6, "activation": "tanh" }
  ],
  "output_units": 2,
  "output_activation": null
}
```

| Field               | Rule                                              |
|---------------------|---------------------------------------------------|
| `name`              | 1–80 characters                                   |
| `input_units`       | integer 1–16                                      |
| `hidden_layers`     | 0–6 layers; `units` 1–32, `activation` `relu` \| `sigmoid` \| `tanh` |
| `output_units`      | integer 1–10                                      |
| `output_activation` | `relu` \| `sigmoid` \| `tanh` \| `null` (raw outputs) |

Unknown fields are rejected.

-   `201 Created` returns `{"model_id": "edu_…", "graph": <ModelGraph>}`.
    The graph has one `Input` node, then a `Linear` node per layer, with
    ids `input`, `hidden_1` … `hidden_n`, and `output`. Each layer
    carries `in_features`, `out_features`, and `activation` attributes
    and a parameter count of `in × out + out`. The model metadata
    includes `educational.parameter_count`, the total.
-   `422` returns `validation_error` with one `details` entry per
    invalid field, e.g. `body.hidden_layers.0.units`.

An optional `seed` (default 0) fixes weight initialization, so the same
config always produces the same network.

**Capabilities:** each educational graph is backed by a PyTorch network,
so `graph`, `inference`, `training`, `activations`, and `gradients` are
`true`; `attention` is `false`.

## `GET /models/{model_id}/graph`

Returns the stored graph (`200`), or `404 not_found` for unknown ids.
`model_id` must match `^[A-Za-z0-9_-]{1,64}$`.

### Storage limitation

Models live in an in-memory repository: per process, lost on restart,
and capped at the 64 most recently used models. Older ones are evicted
and then return `404`; evicting a model cancels its training. The
Architecture Studio creates a new model when a valid edit settles, and
reuses the current model (keeping trained weights) when its
configuration is unchanged.

## `POST /models/{model_id}/forward`

Runs a forward pass and returns the values the model actually computes.

``` json
{ "inputs": [[0.5, -1.0]], "capture": ["hidden_1"] }
```

-   `inputs`: 1–64 rows, each with exactly one value per input unit;
    values must be finite and within ±1,000,000.
-   `capture` (optional): layer ids to return; omit for all layers.

The response has `outputs` (after the output activation), `logits`
(before it), `predictions` (argmax, or `logit > 0` for one output
unit), and `activations[layer] = {pre, post}`, where `pre = W·x + b` and
`post` is after the layer's activation. For the `input` layer, `pre` is
`null` and `post` echoes the inputs. Wrong row lengths or unknown layers
return `422` naming the field (e.g. `body.inputs.1`).

## `GET /models/{model_id}/tensors/{tensor_id}`

Bounded access to weights, biases, and gradients. Tensor ids for
educational models: `{layer}.weight`, `{layer}.bias`, and the same with
`.grad` appended, e.g. `hidden_1.weight.grad`. For imported models:
the `tensor_id` listed in each operator's
`metadata.onnx.initializers` (`t0`, `t1`, …).

-   Query: `offset` (≥ 0) and `limit` (1–4096, default 4096).
-   Returns `shape`, `dtype`, `count`, a `summary`
    (min/max/mean/std over the finite values, or `null` if there are
    none), `non_finite_count`, `values` (row-major from `offset`; NaN and
    infinities are `null` because JSON cannot represent them), and
    `truncated`.
-   Imported tensors stored in external files return `404` (their
    values were not uploaded); string tensors return
    `400 unsupported_tensor`.
-   A weight has shape `[target units, source units]`:
    `weight[j][i]` is the connection from source unit `i` to target unit
    `j`, at flat index `j × source_units + i`.
-   Gradients are a snapshot from the last batch of the latest training
    epoch. Before any training they return `404` ("Train the model
    first").

## `POST /models/{model_id}/training`

Starts a training job and returns `202` with `TrainingJobInfo`
(`job_id`, `model_id`, `status`, `epoch`, `total_epochs`, `config`).
Body (`shared/schemas/training-config.schema.json`, all optional):

| Field              | Default | Range / values                          |
|--------------------|---------|-----------------------------------------|
| `dataset`          | `moons` | `moons`, `circles` (2 classes), `spiral` (one arm per output) |
| `samples`          | 400     | 50–2000                                 |
| `noise`            | 0.1     | 0–1 (Gaussian std added to points)      |
| `validation_split` | 0.2     | 0.1–0.5                                 |
| `epochs`           | 100     | 1–500                                   |
| `batch_size`       | 32      | 1–512                                   |
| `learning_rate`    | 0.03    | > 0, ≤ 1                                |
| `optimizer`        | `adam`  | `adam`, `sgd` (momentum 0.9)            |
| `seed`             | 0       | data generation and batch order         |
| `capture_layers`   | `[]`    | ≤ 8 layer ids to stream activations for |
| `epoch_delay_ms`   | 0       | 0–500 pause per epoch, for watching     |

Datasets are 2D and deterministic for a given seed. Networks with more
than 2 inputs get extra uninformative noise features; a single-input
network sees only x. One output unit trains a single-logit binary
classifier (BCE); more outputs use cross-entropy over that many
classes. Training uses the output layer's raw values (logits). The
configured output activation affects forward-pass `outputs` only.

Errors: `409 conflict` if the model is already training; `429
too_many_requests` when the server is running its maximum number of
jobs (`NEUROVISION_MAX_TRAINING_JOBS`, default 2); `422` for invalid
settings, unknown capture layers, or a dataset whose class count does
not fit the output layer.

Jobs run in a background thread pool, never on the event loop. Weights
are updated under a per-model lock, so concurrent forward passes and
tensor reads never see a half-applied step. If the loss becomes NaN or
infinite, the job fails with a message to lower the learning rate.

## `GET /models/{model_id}/metrics`

`{model_id, job, history}`: the latest job's info and per-epoch
`{epoch, loss, accuracy, val_loss, val_accuracy}`.

## `GET /training/{job_id}` and `POST /training/{job_id}/cancel`

Job status, and a cancellation request (`202`). Cancellation takes
effect at the next batch boundary (or immediately during an epoch
delay) and ends with a `training.cancelled` event.

## `WS /training/{job_id}/events`

Streams `shared/schemas/training-event.schema.json` events:
`training.started`, one `training.step` per epoch (metrics, plus
activations of the captured layers for the first validation example),
then exactly one of `training.completed`, `training.cancelled`, or
`training.failed`. On connect the server replays every event so far,
then streams live ones, so a late or reconnecting client misses
nothing. The server closes the socket after the final event.

-   Unknown job: the socket is accepted, then closed with code `4404`.
-   Browser `Origin` headers must be in `NEUROVISION_CORS_ORIGINS`
    (CORS does not cover WebSockets); otherwise the connection is
    refused.

-   External runs (ids `run_…`) stream on this same socket; see below.

## External runs

Training scripts outside NeuroVision stream their own metrics and
activations into a run. The Python client in `sdk/` wraps these
endpoints; see [live-runs.md](live-runs.md). Bodies follow
`shared/schemas/external-run.schema.json`.

### `POST /runs`

``` json
{ "name": "resnet18 / lr 3e-4", "total_epochs": 20, "model_id": null }
```

`name` 1–80 characters, `total_epochs` 1–1,000,000, optional `model_id`
of a model already in NeuroVision (`422` if unknown). `201 Created`
returns `{"run": <RunInfo>, "write_token": "…"}`. The token is returned
only here. `429` when 8 runs are already active.

### `POST /runs/{run_id}/events`

Requires `Authorization: Bearer <write_token>` (`401` missing, `403`
wrong). The body is an `EventBatch`, up to 32 steps and 2 MB (`413`):

``` json
{"events": [{"epoch": 1, "step": 23,
             "metrics": {"loss": 0.91, "val_loss": 0.84},
             "selected_activations": {"hidden_1": [0.0, 0.41, 1.07]}}]}
```

Each step follows the `training.step` bounds (≤ 16 finite metrics,
≤ 8 layers × 256 finite values). Steps must increase and epochs must
not decrease across the run (`422` with the offending field). The whole
batch is validated before any of it is published. An empty `events`
list is a heartbeat. `202` returns `{"accepted", "steps_logged"}`.
`409` once the run has finished or has logged 10,000 steps.

### `POST /runs/{run_id}/finish`

Same token. `{"status": "completed"}` publishes `training.completed`
with the last logged metrics; `{"status": "failed", "message": "…"}`
publishes `training.failed` (message ≤ 500 characters).

### `GET /runs` and `GET /runs/{run_id}`

`RunInfo` for each retained run, newest first: `run_id`, `name`,
`model_id`, `status` (`running` | `completed` | `failed`),
`total_epochs`, `epoch`, `step`, `steps_logged`, `created_at`, and
`last_event_at` (Unix seconds). Tokens are never included. A run with
no events or heartbeats for 5 minutes is marked `failed` when the API
is next accessed.

## `POST /import/onnx?filename=…`

Imports an ONNX model for inspection. Send the file as the raw request
body (`Content-Type: application/octet-stream`); `filename` must end in
`.onnx`. Returns `201` with `{model_id, graph, report}`, where `report`
has the file name and size, operator / edge / initializer counts, the
initializer value count, external tensor count, opsets, unsupported
operators, and warnings (`shared/schemas/import.schema.json`).

The file is parsed in memory and never executed; external data paths it
references are never opened. Errors:

-   `413 payload_too_large`: larger than `NEUROVISION_MAX_UPLOAD_BYTES`
    (default 64 MB), checked from `Content-Length` and while streaming.
-   `422 validation_error`: wrong extension or empty body.
-   `422 invalid_model`: not parseable as ONNX, no graph, or more than
    5,000 operators.

The graph's capabilities report `inference` and `activations` as `true`
when the model can be run (see `POST /models/{model_id}/run`).
`report.inputs` lists the inputs a run must provide, and
`report.inference_blockers` explains why a model cannot run. `training`
and `gradients` are always `false`. Forward, training, and metrics
requests for imported models return `400 capability_unavailable`.

## `POST /models/{model_id}/run`

Runs an imported model in ONNX Runtime, inside an isolated worker process
(the API process never loads the runtime). Body
(`shared/schemas/onnx-run.schema.json`):

``` json
{
  "inputs": { "image": { "shape": [1, 3, 32, 32], "fill": "random", "seed": 0 } },
  "capture": ["node_4", "node_9"]
}
```

-   `inputs`: one entry per declared input. `shape` must match the
    declared rank and every fixed dimension; named or unknown
    dimensions take 1–100,000. The total across inputs is at most
    1,000,000 values. Give `values` (row-major, finite; whole numbers in
    range for integer inputs), or use `fill`: `random` (seeded normal
    for floats, 0–9 for integers), `zeros`, `ones`, or `range`
    (−1 to 1).
-   `capture` (optional): operator node ids whose outputs to return.
    Omitted, every operator is captured for graphs up to 200 operators.

Returns `{model_id, run_id, retained, inputs, runtime, elapsed_ms,
outputs, intermediates, notes}`. `run_id` identifies the run's tensors
for the map views below; `retained` is `false` when they were too large
to keep; `inputs` gives each fed input's concrete shape.
Each tensor has `name`, producing `node_id`, `shape`, `dtype`, `count`,
`summary` (finite values), `non_finite_count`, at most 4,096 `values`,
and `truncated`. `elapsed_ms` is model execution only; `notes` mention
worker startup and model loading.

Errors: `422 validation_error` (inputs that don't match the model, with
the field), `422 inference_unsupported` (the import's blockers),
`422 inference_failed` (ONNX Runtime rejected the model or inputs; its
first message line), `504 inference_timeout` (over
`NEUROVISION_INFERENCE_TIMEOUT_SECONDS`, default 10; the worker is
restarted), `500 inference_crashed` (the worker died; it restarts on the
next run), `429` (another run in progress), and `400
capability_unavailable` for educational models (use `/forward`).
Imported models share the in-memory repository, which also has a 512 MB
budget and evicts the least recently used models.

Supported and unsupported cases are listed in
[importing-onnx.md](importing-onnx.md).

## `GET /models/{model_id}/maps` and `GET /models/{model_id}/grids`

Bounded 2D views of one tensor from a recent run
(`shared/schemas/tensor-view.schema.json`). Query parameters:

-   `run_id`: from the run response (`retained` must have been `true`).
-   `node_id`: the operator (or `input_*` node for a graph input).
-   `output` (default 0): which of the node's outputs.
-   `fixed` (repeated): leading indices. For `maps`, one per dimension
    before the last two (e.g. `fixed=0&fixed=3` is batch 0, channel 3 of
    a 4D tensor). For `grids`, one per dimension before the last three.

`maps` returns `{rows, cols, stride, values, min, max,
non_finite_count, …}`: the last two dimensions, shrunk by `stride` ×
`stride` block averages when over 65,536 cells. `grids` returns one item
per index along the third-to-last dimension (at most 256, `items_total`
says how many exist), each at most 24 × 24 cells, plus the overall
`min`/`max` for a shared color scale. NaN/∞ values are `null` and left
out of averages.

Errors: `410 run_expired` (the run is no longer kept: run again),
`422 validation_error` naming `query.node_id`, `query.output` or
`query.fixed`, `504`/`429` as for runs, and `400
capability_unavailable` for educational models.

Imports report likely attention in `report.attention`: `{node_id, name,
tensor, shape, layout}`, found by pattern (a `Softmax` over the last axis
of a square `[..., S, S]` tensor).

## `GET /import/onnx/samples` and `POST /import/onnx/samples/{name}`

Lists, and imports, small sample models (`mlp`, `cnn`, `residual`,
`custom-op`, `attention`) built in memory. They go through the same importer as
uploads. Their weights are seeded random values, so they show structure
and tensors, not trained behavior.
