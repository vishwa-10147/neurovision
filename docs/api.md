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

**Capabilities:** the graph reports `graph: true` only. No PyTorch model
exists yet, so inference, training, activations, and gradients are
`false` until Milestone 4.

## `GET /models/{model_id}/graph`

Returns the stored graph (`200`), or `404 not_found` for unknown ids.
`model_id` must match `^[A-Za-z0-9_-]{1,64}$`.

### Storage limitation

Models live in an in-memory repository: per process, lost on restart,
and capped at the 64 most recently used models. Older ones are evicted
and then return `404`. The Architecture Studio creates a new model each
time a valid edit settles, so it does not depend on old ids.
