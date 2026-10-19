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

Bounded access to weights, biases, and gradients. Tensor ids:
`{layer}.weight`, `{layer}.bias`, and the same with `.grad` appended,
e.g. `hidden_1.weight.grad`.

-   Query: `offset` (≥ 0) and `limit` (1–4096, default 4096).
-   Returns `shape`, `dtype`, `count`, a `summary`
    (min/max/mean/std over the whole tensor), `values` (row-major, from
    `offset`), and `truncated`.
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
