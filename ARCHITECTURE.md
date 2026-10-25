# NeuroVision Architecture

## 1. Purpose

This document defines the system boundaries, data contracts, processing
paths, and engineering constraints for NeuroVision.

NeuroVision has two main model sources: - Educational networks created
and trained within the application. - Imported models, initially ONNX,
for graph and tensor inspection.

Both should feed a shared normalized graph model. The visualization
layer must not depend directly on PyTorch or ONNX internals.

## 2. High-level architecture

``` mermaid
flowchart TB
    User --> WebApp
    subgraph Frontend
      WebApp[React application shell]
      Studio[Architecture Studio]
      Graph2D[2D renderer]
      Graph3D[3D renderer]
      Inspector[Inspector and metrics]
      Store[Shared client state]
      WebApp --> Studio
      WebApp --> Graph2D
      WebApp --> Graph3D
      WebApp --> Inspector
      Studio --> Store
      Graph2D --> Store
      Graph3D --> Store
      Inspector --> Store
    end
    subgraph Backend
      API[FastAPI REST API]
      WS[WebSocket event service]
      Normalizer[Graph normalizer]
      Educational[Educational network service]
      Importer[ONNX import service]
      Training[Training job manager]
      Tensor[Bounded tensor access]
      API --> Educational
      API --> Importer
      API --> Normalizer
      API --> Tensor
      Educational --> Training
      Importer --> Normalizer
      Training --> WS
    end
    Store <--> API
    Store <--> WS
    Educational --> PyTorch[PyTorch]
    Training --> PyTorch
    Importer --> ONNX[ONNX]
```

## 3. Component responsibilities

### Frontend

-   `app/`: application shell, sections, and the shared model view.
-   `features/architecture-studio/`: model configuration, editing,
    forward passes, and saved projects.
-   `features/visualization-2d/`: 2D node, edge, and signal rendering.
-   `features/visualization-3d/`: Three.js scene, camera, selection, and
    rendering.
-   `features/training-lab/`: training controls, progress, charts, and
    event display.
-   `features/model-import/`: upload, parsing status, ONNX runs, and
    graph and tensor exploration (feature maps, filters, attention).
-   `features/live-runs/`: external runs from users' own scripts: charts,
    health, system telemetry, controls, custom visuals, and the live graph.
-   `features/classical/`: the Classical ML Lab (scikit-learn algorithms).
-   `features/neural-lab/`: launching bundled training scripts.
-   `features/planned/`: placeholders for planned sections. Learning Mode
    (contextual explanations and guided exercises) is planned.
-   `components/inspector/`: selected graph element details.
-   `stores/`: shared state and selected IDs.
-   `api/`: typed API client.
-   `types/`: frontend types and validated schema representations.

### Backend

-   `api/routes/`: HTTP and WebSocket endpoints.
-   `schemas/`: Pydantic request/response models.
-   `services/educational_graph.py`: build the normalized graph for an
    educational dense network.
-   `services/onnx_importer.py`: parse ONNX models into the normalized
    graph without executing them.
-   `services/onnx_runtime.py`: run imported ONNX models in an isolated,
    time- and memory-limited worker process (`ml/onnx_worker.py`).
-   `services/inference.py`: forward passes and bounded tensor access for
    educational models; `services/tensors.py`: bounded tensor summaries
    and slices shared by every model source.
-   `services/model_repository.py`: storage for created and imported
    models, behind an interface.
-   `services/training_jobs.py`: background training jobs;
    `services/event_stream.py`: replayable per-run event history shared
    by jobs and external runs.
-   `services/external_runs.py` and `services/run_health.py`: runs
    streamed from users' scripts, and their health alerts.
-   `services/project_store.py`: atomic, versioned local project files.
-   `services/classical.py` and `services/lab.py`: the Classical ML Lab
    and the Neural Lab.
-   `ml/educational_network.py`: configurable educational PyTorch
    network with explicit activation capture.
-   `ml/trainer.py`: training loops, cancellation, and metrics.
-   `ml/datasets.py`: deterministic sample datasets.
-   `ml/tensor_maps.py`: bounded 2D views of tensors.
-   `core/config.py`: environment-backed settings, including upload and
    job limits.
-   `core/errors.py`: the shared error envelope.

Do not place long-running training loops inside request handlers or
block the ASGI event loop.

## 4. Normalized graph contract

The normalized graph is the core integration boundary. It should be
versioned and documented in `shared/schemas/model-graph.schema.json`.

Conceptual shape:

``` json
{
  "schema_version": "1.0",
  "model": {
    "id": "model_123",
    "name": "Example Dense Network",
    "source_format": "educational",
    "capabilities": {
      "graph": true,
      "inference": true,
      "training": true,
      "activations": true,
      "gradients": true,
      "attention": false
    }
  },
  "nodes": [
    {
      "id": "layer_1",
      "name": "Hidden Layer 1",
      "op_type": "Linear",
      "input_tensors": ["input_0"],
      "output_tensors": ["tensor_1"],
      "input_shapes": [[1, 4]],
      "output_shapes": [[1, 8]],
      "attributes": {
        "activation": "relu"
      },
      "parameter_summary": {
        "parameter_count": 40
      }
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": "input_layer",
      "target": "layer_1",
      "tensor_name": "input_0"
    }
  ]
}
```

This is an illustrative contract, not a final schema. The actual schema
must define required/optional fields, identifiers, enums, and validation
rules.

### Contract rules

-   Node and edge IDs must be stable within a model version.
-   Graph topology must be separate from UI layout coordinates.
-   Tensor data should be referenced, not embedded in graph responses.
-   Unknown operators must be preserved as inspectable nodes.
-   Shapes may be unknown or dynamic; do not fabricate dimensions.
-   Model capabilities must describe real supported operations.
-   A graph imported for inspection is not automatically executable or
    trainable.
-   Keep source-specific metadata in a namespaced metadata field when it
    cannot be normalized cleanly.

## 5. 2D and 3D renderer contract

Both renderers consume the same graph and shared selection state.

### 2D

Use React Flow for editing and high-level architecture graphs. Use a
custom SVG or Canvas view for detailed neuron-level diagrams when React
Flow's node model would be too heavy.

### 3D

Use React Three Fiber / Three.js for graph nodes, edges, camera
controls, and animations. Use instancing and batching for repeated
geometry. Add levels of detail: 1. Model-level overview. 2.
Layer/operator-level graph. 3. Selected layer details. 4. Individual
neurons or selected tensor/connection details.

Do not create one 3D object per parameter for a large model. Filter,
aggregate, sample, or drill down.

### Shared selection

The shared selection state should hold semantic IDs such as
`selectedNodeId` or `selectedEdgeId`, not renderer-specific object
references. Each renderer maps those IDs to its own visual objects.

## 6. Training and event flow

1.  Client submits a validated training configuration.
2.  API creates a training job and returns a job ID.
3.  Job manager runs the training loop outside the request event loop.
4.  Training loop records metrics and captures only requested
    visualization values.
5.  WebSocket streams compact events.
6.  Client updates charts, selected neuron data, and signal animation.
7.  Client can request cancellation; the backend marks the job as
    cancelled and performs cleanup.

Example event:

``` json
{
  "schema_version": "1.0",
  "event_type": "training.step",
  "job_id": "job_123",
  "step": 10,
  "metrics": {
    "loss": 0.42,
    "accuracy": 0.81
  },
  "selected_activations": {
    "layer_1": [0.12, 0.91, 0.04]
  }
}
```

Event payloads must be bounded. Do not stream all weights, gradients,
and activations on every step. Support selected-layer capture and
sampling.

## 7. Model import flow

1.  Receive upload into a controlled temporary location.
2.  Enforce size limits and validate the file.
3.  Parse with the ONNX library without automatically executing the
    model.
4.  Extract graph topology, operators, tensor names, shapes,
    initializers, and metadata.
5.  Normalize the graph.
6.  Return a capability report and import warnings.
7.  Allow graph inspection and bounded tensor queries.
8.  If ONNX Runtime execution is later enabled, run it through a
    separate execution boundary with limits and isolation appropriate to
    the deployment.

Do not use unsafe pickle deserialization for untrusted files. Do not
treat an extension check as sufficient validation.

## 8. API design

Base path: `/api/v1`.

  ------------------------------------------------------------------------------------------
  Endpoint                                   Method                  Purpose
  ------------------------------------------ ----------------------- -----------------------
  `/health`                                  GET                     Service health

  `/models/educational`                      POST                    Create educational
                                                                     network

  `/models/{model_id}/graph`                 GET                     Get normalized graph

  `/models/{model_id}/forward`               POST                    Execute educational
                                                                     forward pass

  `/models/{model_id}/training`              POST                    Start a training job

  `/models/{model_id}/metrics`               GET                     Retrieve metrics

  `/import/onnx`                             POST                    Import and inspect ONNX
                                                                     model

  `/models/{model_id}/tensors/{tensor_id}`   GET                     Bounded tensor
                                                                     summary/slice

  `/training/{job_id}/events`                WS                      Training events
  ------------------------------------------------------------------------------------------

Define request and response schemas before implementing each endpoint.
Use consistent error structures, correct HTTP status codes, and explicit
validation.

## 9. State and persistence

### Initial development

-   Keep graph state in memory or local development storage behind an
    interface.
-   Avoid adding a database before there is a defined persistence
    requirement.
-   Keep uploaded files outside version control.

### Later persistence

-   Persist project definitions and user preferences.
-   Add PostgreSQL for multi-user projects and experiment metadata if
    required.
-   Store model binaries in controlled object storage if scale or
    deployment needs justify it.
-   Add ownership and access checks before exposing multi-user project
    APIs.

## 10. Security and reliability

-   Validate file type, file size, parsing errors, tensor slice bounds,
    and architecture limits.
-   Do not expose arbitrary filesystem paths through API parameters.
-   Do not automatically execute uploaded models.
-   Do not load untrusted Python pickle files.
-   Apply request timeouts and resource limits.
-   Keep training jobs isolated from the API request lifecycle.
-   Implement cancellation and cleanup.
-   Configure CORS explicitly per environment.
-   Store secrets only in environment variables.
-   Avoid returning internal stack traces, secrets, or server paths to
    clients.
-   Log job/model IDs and safe error context, not sensitive tensor data
    by default.

## 11. Performance strategy

-   Use level-of-detail rendering.
-   Use instancing for repeated 3D nodes.
-   Aggregate or filter dense edges.
-   Limit tensor slice sizes.
-   Send compact WebSocket messages.
-   Throttle UI updates to a reasonable display rate.
-   Avoid recomputing the full scene for every training event.
-   Cache normalized graphs and derived layouts when safe.
-   Measure performance with representative graph sizes before claiming
    scalability.

## 12. Testing boundaries

-   Unit tests for graph normalization and schema validation.
-   Numerical tests for forward pass and training.
-   API tests for health, model creation, graph retrieval, and invalid
    requests.
-   Upload tests for malformed and oversized ONNX files.
-   Tests for unknown operators and dynamic tensor shapes.
-   Frontend tests for synchronized selection and 2D/3D switching.
-   Integration tests for creating a model, running a forward pass, and
    displaying returned values.

## 13. Architecture decision records to maintain

Record decisions when they materially affect the system: - Why the
normalized graph schema is versioned. - Which graph renderer handles
each scale. - How training jobs are scheduled and cancelled. - Which
model formats are supported and under what restrictions. - When
persistent storage is introduced. - What resource limits apply to
uploaded models and tensor access.
