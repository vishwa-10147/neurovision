# NeuroVision Development Plan

## Objective

Deliver an advanced platform for educational neural-network exploration
and real-model inspection, with synchronized 2D and 3D views, a PyTorch
learning engine, and ONNX graph import.

Work milestone by milestone. Each milestone must leave the repository in
a runnable, testable state.

## Principles

1.  Build a vertical slice before adding breadth.
2.  Define the shared graph contract before implementing both renderers.
3.  Use real computed values whenever the UI claims to show model
    behavior.
4.  Distinguish graph inspection from inference, training, and gradient
    support.
5.  Add persistence and deployment complexity only when justified.
6.  Update docs and tests with each meaningful change.

## Milestone 0 --- Repository and development environment

### Tasks

-   [x] Create repository and root documentation.
-   [x] Set up `frontend/` with React, TypeScript, and Vite.
-   [x] Set up `backend/` with Python 3.12 and FastAPI.
-   [x] Add `.gitignore` and `.env.example`.
-   [x] Add frontend and backend formatting/lint commands.
-   [x] Add basic health endpoint test.
-   [x] Verify both services run locally.

### Acceptance criteria

-   Frontend starts using documented commands.
-   Backend starts using documented commands.
-   `GET /api/v1/health` returns a documented response.
-   Automated health test passes.
-   No secrets or local model files are committed.

## Milestone 1 --- Application shell and shared contracts

### Tasks

-   [x] Build navigation for Dashboard, Architecture Studio, Training
    Lab, Model Explorer, Learning Mode.
-   [x] Add backend connection status.
-   [x] Define model graph schema and event schema.
-   [x] Create typed frontend API client.
-   [x] Add loading, empty, and error states.
-   [x] Add request/response validation and tests.

### Acceptance criteria

-   The UI reports backend availability correctly.
-   Graph schema is documented and validated.
-   API errors appear as actionable messages.

## Milestone 2 --- 2D architecture studio

### Tasks

-   [x] Define editable dense-network configuration.
-   [x] Add input, hidden, and output layer controls.
-   [x] Support safe layer width limits.
-   [x] Add ReLU, sigmoid, and tanh configuration.
-   [x] Render architecture in 2D.
-   [x] Implement selection and inspector.
-   [x] Add zoom, pan, fit-to-view, and reset.
-   [x] Display parameter counts when correct.
-   [x] Add validation for invalid configurations.

### Acceptance criteria

-   Users can build a small dense network.
-   Node/edge selection updates the inspector.
-   Invalid configurations are rejected with clear messages.
-   The layout is derived from semantic graph data and stored
    separately.

## Milestone 3 --- 3D explorer

### Tasks

-   [ ] Render the same normalized graph in Three.js.
-   [ ] Add orbit, zoom, pan, and camera reset.
-   [ ] Add selected-node and selected-edge highlighting.
-   [ ] Synchronize selection with 2D view and inspector.
-   [ ] Add 2D/3D switching without losing state.
-   [ ] Add initial performance safeguards and detail limits.
-   [ ] Respect reduced-motion settings for nonessential animation.

### Acceptance criteria

-   The same model can be explored in both views.
-   Selection remains synchronized.
-   The 3D view remains usable on a representative small/medium graph.
-   Large graph limitations are explicit.

## Milestone 4 --- Educational computation engine

### Tasks

-   [ ] Implement configurable PyTorch feedforward network.
-   [ ] Add deterministic synthetic classification dataset.
-   [ ] Implement input validation and forward-pass endpoint.
-   [ ] Capture actual activations for selected layers.
-   [ ] Add training loop with validation metrics.
-   [ ] Capture selected weights and gradients.
-   [ ] Add loss and accuracy charts.
-   [ ] Add WebSocket event schema and streaming.
-   [ ] Add cancellation and cleanup.
-   [ ] Add numerical and API tests.

### Acceptance criteria

-   A sample network runs a forward pass.
-   Training produces finite loss values and measurable progress on the
    sample task.
-   Displayed activations originate from the actual model computation.
-   Training events are bounded and do not stream the entire model
    state.
-   A training job can be cancelled safely.

## Milestone 5 --- ONNX import and graph inspection

### Tasks

-   [ ] Add controlled upload endpoint and size limits.
-   [ ] Validate ONNX files and return safe parse errors.
-   [ ] Extract graph nodes, operators, edges, metadata, initializers,
    and known shapes.
-   [ ] Normalize the graph into the shared schema.
-   [ ] Preserve unknown operators with warnings.
-   [ ] Add capability report.
-   [ ] Add bounded tensor summaries and slices.
-   [ ] Add tests for malformed files, dynamic shapes, unknown
    operators, and oversized requests.
-   [ ] Document supported and unsupported cases.

### Acceptance criteria

-   A supported ONNX model imports and appears in both graph views.
-   The importer does not automatically execute the model.
-   Unsupported operators are visible and not silently dropped.
-   Tensor endpoints enforce bounds and response limits.

## Milestone 6 --- Real-model inference

### Tasks

-   [ ] Add ONNX Runtime only after graph inspection is stable.
-   [ ] Define supported input tensor types/shapes.
-   [ ] Validate inference inputs.
-   [ ] Add inference time and resource limits.
-   [ ] Display output tensors and relevant intermediate values where
    supported.
-   [ ] Isolate execution and document deployment restrictions.
-   [ ] Test invalid input shapes and unsupported operations.

### Acceptance criteria

-   A compatible ONNX model can be executed with validated inputs.
-   Execution failures are returned safely.
-   Graph import remains usable even when inference is unsupported.

## Milestone 7 --- Advanced architectures

### CNN

-   [ ] Visualize convolution, pooling, and feature maps.
-   [ ] Show tensor dimensions and filter parameters.
-   [ ] Support selected-channel views and bounded data access.

### Transformers

-   [ ] Visualize tokens, embeddings, attention heads, and attention
    matrices.
-   [ ] Make model-specific capability limitations explicit.
-   [ ] Avoid rendering full attention tensors without filtering.

### Acceptance criteria

-   Each new visualization has a defined data contract, tests, and clear
    limitations.
-   Existing dense-network workflows continue to work.

## Milestone 8 --- Persistence and project management

### Tasks

-   [ ] Save/load architecture definitions.
-   [ ] Add versioned project format.
-   [ ] Add experiment metadata and metrics history.
-   [ ] Introduce PostgreSQL only if multi-user persistence is required.
-   [ ] Add file ownership and access control if user accounts are
    added.
-   [ ] Add controlled model storage if needed.

### Acceptance criteria

-   Projects can be saved and reopened without data loss.
-   Schema upgrades are handled explicitly.
-   Uploaded files remain protected by access controls.

## Milestone 9 --- Production readiness

### Tasks

-   [ ] Add Dockerfiles and Compose configuration.
-   [ ] Configure production CORS and environment variables.
-   [ ] Add request limits, timeouts, health checks, and structured
    logs.
-   [ ] Add frontend production build and backend startup checks.
-   [ ] Add deployment instructions.
-   [ ] Perform security review of uploads and model execution.
-   [ ] Measure rendering performance with representative models.
-   [ ] Document backups, data retention, and supported model formats.

### Acceptance criteria

-   A clean environment can reproduce the documented deployment.
-   No development secrets are required.
-   Tests, linting, type checks, and production build pass.
-   Known limitations and resource requirements are documented.

## Suggested issue breakdown

Use one issue per small vertical slice: - `NV-001` Repository scaffold
and health endpoint - `NV-002` Shared graph schema - `NV-003` Frontend
API client and health status - `NV-004` Dense-network editor - `NV-005`
2D graph renderer - `NV-006` Selection inspector - `NV-007` 3D graph
renderer - `NV-008` Shared selection state - `NV-009` PyTorch model
factory - `NV-010` Forward-pass endpoint - `NV-011` Activation capture -
`NV-012` Training jobs and metrics - `NV-013` WebSocket event stream -
`NV-014` ONNX upload and normalization - `NV-015` Tensor inspection -
`NV-016` ONNX Runtime inference - `NV-017` Project save/load - `NV-018`
Docker and deployment

## Release gates

### Alpha

-   App shell, shared graph schema, 2D and 3D dense network views.
-   A real PyTorch forward pass and basic training.

### Beta

-   ONNX graph inspection, tensor inspection, tests, and documented
    limits.
-   Reliable error handling and bounded visualization.

### Production candidate

-   Persistence, deployment, security review, resource limits,
    monitoring, and reproducible build.
-   No placeholder presented as a completed feature.

## Definition of done for every milestone

-   Code is integrated end-to-end.
-   Relevant automated tests pass.
-   Type checks, lint, and build are run where configured.
-   Errors and empty states are handled.
-   Documentation reflects actual behavior.
-   Known limitations are recorded.
-   The next milestone has clear acceptance criteria.
