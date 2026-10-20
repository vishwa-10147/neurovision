# Shared contracts

NeuroVision's frontend and backend exchange data through versioned
contracts. The **Pydantic models in `backend/app/schemas/` are the source
of truth**. The generated JSON Schema files in `shared/schemas/` are the
published contract. The Zod schemas in `frontend/src/types/` and
`frontend/src/api/client.ts` mirror them for runtime validation in the
browser.

| Contract        | Pydantic                          | JSON Schema                                | Zod                              |
|-----------------|-----------------------------------|--------------------------------------------|----------------------------------|
| Model graph     | `app/schemas/graph.py`            | `shared/schemas/model-graph.schema.json`   | `src/types/graph.ts`             |
| Training events | `app/schemas/events.py`           | `shared/schemas/training-event.schema.json`| `src/types/events.ts`            |
| Dense config    | `app/schemas/architecture.py`     | `shared/schemas/dense-network-config.schema.json` | `src/types/architecture.ts` |
| Training config | `app/schemas/training.py`         | `shared/schemas/training-config.schema.json` | `src/types/training.ts`        |
| Forward pass    | `app/schemas/training.py`         | `shared/schemas/forward.schema.json`       | `src/types/training.ts`          |
| Tensor slice    | `app/schemas/training.py`         | `shared/schemas/tensor.schema.json`        | `src/types/training.ts`          |
| ONNX import     | `app/schemas/imports.py`          | `shared/schemas/import.schema.json`        | `src/types/imports.ts`           |
| ONNX run        | `app/schemas/inference.py`        | `shared/schemas/onnx-run.schema.json`      | `src/types/imports.ts`           |
| Tensor views    | `app/schemas/inference.py`        | `shared/schemas/tensor-view.schema.json`   | `src/types/maps.ts`              |
| Error envelope  | `app/schemas/errors.py`           | `shared/schemas/error.schema.json`         | `src/api/client.ts`              |
| Health          | `app/schemas/health.py`           | `shared/schemas/health.schema.json`        | `src/api/health.ts`              |

## Keeping them in sync

-   After changing a Pydantic model, regenerate the JSON Schema files
    from `backend/` with the virtual environment active:

    ``` powershell
    python -m app.schemas.export
    ```

    `tests/test_schemas.py` fails if a committed schema file does not
    match the models.
-   `shared/schemas/examples/` holds example documents and
    `invalid-cases.json`. The backend tests (`tests/test_schemas.py`) and
    the frontend tests (`src/types/contracts.test.ts`) validate the same
    files. Every valid example must pass on both sides, and every invalid
    case must be rejected on both sides. When you change a rule, add a
    case to `invalid-cases.json`.

## Model graph (schema version 1.0)

A normalized, renderer-independent description of a model's computation
graph.

-   **`model`**: `id`, `name`, `source_format` (`educational` |
    `onnx`), `capabilities`, and namespaced `metadata`.
-   **`capabilities`**: six required booleans (`graph`, `inference`,
    `training`, `activations`, `gradients`, `attention`). They describe
    what is actually supported. An imported ONNX graph is inspection-only
    until inference is implemented.
-   **`nodes`**: `id`, `name`, `op_type`, input/output tensor names,
    optional shapes, `attributes`, an optional `parameter_summary`,
    `unsupported`, and `metadata`.
-   **`edges`**: `id`, `source` node id, `target` node id, and an
    optional `tensor_name`.
-   **`warnings`**: import warnings, such as unsupported operators.

### Rules

-   **Shapes are never invented.** A dimension is a non-negative integer,
    a symbolic name (e.g. `"batch"`, `"N"`), or `null` when unknown. A
    whole shape is `null` when even its rank is unknown.
-   **No layout data.** Unknown fields are rejected, including `x`/`y`
    coordinates. Renderers compute and store layout separately.
-   **No tensor data.** Weights and activations are referenced by tensor
    name and served by bounded tensor endpoints (planned).
-   **Unknown operators are kept** as nodes with `unsupported: true`,
    usually alongside a `warnings` entry.
-   **Source-specific data** goes in `metadata` under a namespace such as
    `"onnx"`.
-   **Integrity** (checked by both validators, not expressible in JSON
    Schema): node and edge ids are unique, every edge endpoint is an
    existing node, and a non-empty shape list has one entry per tensor.

See `shared/schemas/examples/dense-network.graph.json` and
`onnx-import.graph.json`.

## Training events (schema version 1.0)

These are sent as JSON messages over `WS /api/v1/training/{job_id}/events`
(the endpoint is planned for Milestone 4). `event_type` is the
discriminator:

| `event_type`         | Fields                                                     |
|----------------------|------------------------------------------------------------|
| `training.started`   | `model_id`, `total_epochs` (≥ 1)                           |
| `training.step`      | `epoch`, `step`, `metrics`, `selected_activations`         |
| `training.completed` | `epoch`, `step`, `metrics`                                 |
| `training.cancelled` | `step`                                                     |
| `training.failed`    | `message` (≤ 500 characters)                               |

All events also carry `schema_version` and `job_id`.

### Bounds

Payloads stay small, so the full model state is never streamed.

-   At most 16 metrics per event. Values must be finite numbers.
-   `selected_activations`: at most 8 layers (keyed by graph node id),
    each with at most 256 values. Values are measured, possibly sampled,
    never simulated.

## Error envelope

Every API error response uses this shape:

``` json
{
  "error": {
    "code": "validation_error",
    "message": "The request did not pass validation.",
    "details": [{ "field": "body.width", "message": "Input should be less than or equal to 64" }]
  }
}
```

| HTTP | `code`               |
|------|----------------------|
| 400  | `bad_request`        |
| 404  | `not_found`          |
| 405  | `method_not_allowed` |
| 409  | `conflict`           |
| 413  | `payload_too_large`  |
| 422  | `validation_error`   |
| 500  | `internal_error`     |

Messages are safe to show to users. Validation errors list field
locations but do not echo the submitted values. Server errors return a
generic message and are logged on the server. The frontend turns
envelopes into an `ApiError` with `code`, `details`, and a `hint` for
what to do next, and renders it with `ErrorState`.
