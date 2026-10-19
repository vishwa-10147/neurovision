# NeuroVision --- Master Build Prompt

## Role

Act as a senior full-stack engineer, machine-learning engineer,
visualization specialist, security-minded architect, and technical lead.
Build **NeuroVision**, a maintainable, production-minded web platform
for visually exploring neural networks in both 2D and 3D. The platform
must support educational networks that users can construct and train, as
well as inspection of real trained models imported from supported
formats.

Do not attempt to implement the entire platform in one giant change.
Work in small, testable milestones. Before editing, inspect the
repository and its existing code. Preserve working behavior, follow
existing conventions, and avoid replacing files unnecessarily. If the
repository is empty, scaffold the structure described below.

## Product goal

NeuroVision should help users understand what happens inside neural
networks. Users should be able to: - Create and edit simple
neural-network architectures. - View the same normalized graph in 2D and
3D. - Select a neuron, layer, operator, or connection and inspect its
properties. - Run a forward pass and see real activation values and
signal contributions. - Train educational networks and inspect loss,
metrics, weights, and gradients. - Import supported ONNX models and
explore their computation graphs, operators, tensor shapes, metadata,
and selected tensor values. - Learn concepts through contextual
explanations and guided experiments. - Save and reopen projects as the
persistence layer is added.

The UI should feel like a polished AI laboratory: dark-first but
accessible, clean typography, clear visual hierarchy, responsive layout,
and restrained animation. Do not sacrifice performance or correctness
for decorative effects.

## Core technical decisions

### Frontend

-   React + TypeScript + Vite.
-   React Three Fiber and Three.js for the 3D scene.
-   React Flow (`@xyflow/react`) for layer-level architecture editing
    where appropriate.
-   A custom SVG/Canvas renderer for dense 2D neuron and connection
    views.
-   Zustand for shared client state.
-   Recharts for metrics charts.
-   Zod for validating API responses and shared frontend data.
-   Lucide icons where useful.

### Backend

-   Python 3.12.
-   FastAPI, Pydantic, Uvicorn.
-   PyTorch for educational networks, training, inference, activation
    capture, and gradient capture.
-   NumPy for numerical utilities and small educational calculations.
-   ONNX for parsing and inspecting ONNX graphs.
-   ONNX Runtime as an optional later addition for executing compatible
    imported models.
-   WebSockets for compact, real-time training and visualization events.
-   Pytest and HTTPX for backend tests; Ruff for linting and formatting.

### Repository layout

Use a single repository: - `frontend/` --- React application. -
`backend/` --- FastAPI service and ML modules. - `shared/schemas/` ---
documented graph and event schema definitions. - `docs/` --- project
documentation. - `tests/` --- integration tests and test fixtures. -
root `README.md`, `ARCHITECTURE.md`, `PLAN.md`, `SKILLS.md`,
`PROMPT.md`, `.gitignore`, and eventually `docker-compose.yml`.

Create modules progressively. Do not create dozens of empty files merely
to match a tree. Each module should have a clear responsibility and be
used by the application or be explicitly marked as planned.

## Required architecture

Use one normalized model-graph representation as the contract between
import/training services and both visualizers. A graph should
represent: - Model metadata and capabilities. - Nodes with stable IDs,
names, operator types, attributes, input/output tensor references, and
available shape/type metadata. - Edges with stable IDs and
source/destination references, including tensor names when available. -
Optional parameter metadata and references to separately served tensor
data. - Optional activations, gradients, and weight summaries. - A
capability description indicating what the current model/runtime
supports: graph inspection, inference, training, activations, gradients,
or attention.

Do not place large weight arrays or activation tensors directly in graph
JSON. Expose bounded tensor summaries or slices through API endpoints.
The 2D renderer, 3D renderer, inspector, and metrics panels must consume
the same normalized model state. Keep visualization layout coordinates
separate from semantic graph data.

Do not assume every imported model supports training or gradient
visualization. The UI must show capability-specific actions and clearly
explain unavailable operations.

## Functional requirements

### A. Application shell

-   Provide navigation for Dashboard, Architecture Studio, Training Lab,
    Model Explorer, and Learning Mode.
-   Provide loading, empty, error, and success states.
-   Include a backend connection/health indicator.
-   Use accessible labels, keyboard navigation, sensible focus states,
    and responsive layouts.
-   Keep components small and typed; avoid one enormous component.

### B. Architecture Studio

-   Create a configurable feedforward network with input, hidden, and
    output layers.
-   Initially support fully connected (dense) layers and ReLU, sigmoid,
    and tanh activations.
-   Allow adding/removing hidden layers, changing layer widths within
    safe limits, and changing activations.
-   Validate architecture settings before submitting them to the
    backend.
-   Show parameter counts when they can be calculated correctly.
-   Support 2D/3D view switching without losing model selection.
-   Provide reset-to-example and clear validation messages.

### C. 2D visualization

-   Render layers, neurons/units, and edges.
-   Allow pan, zoom, fit-to-view, and selection.
-   Display selected node/edge details in a synchronized inspector.
-   Encode positive/negative or magnitude differences with a legend when
    weight values are available. Never imply that a color means
    something unless the legend explains it.
-   Use edge filtering, sampling, aggregation, or layer-level views for
    dense graphs.
-   Avoid creating a DOM element for every connection in very large
    models if that causes performance issues.

### D. 3D visualization

-   Use React Three Fiber / Three.js.
-   Support orbit, zoom, pan, camera reset, and clear selection
    feedback.
-   Display neurons or graph operators as nodes and edges as
    connections.
-   Synchronize selected IDs with the 2D renderer and inspector.
-   Animate signal propagation only when real event/value data is
    available.
-   Respect reduced-motion preferences where practical.
-   Use instancing, batching, visibility culling, edge filtering, and
    detail levels for performance.
-   Do not render huge models at full neuron/edge detail by default.

### E. Educational network and training

-   Implement a small feedforward network using PyTorch.
-   Provide a deterministic sample dataset, such as a synthetic
    two-class classification dataset.
-   Support a forward pass, configurable training, and validation
    metrics.
-   Capture actual layer activations and gradients using supported
    PyTorch hooks or explicit model instrumentation.
-   Display loss and accuracy over time.
-   Show training progress and allow a training job to be cancelled
    safely.
-   Stream compact WebSocket events containing job ID, step/epoch,
    metrics, and only the selected visualization data.
-   Do not stream the complete model state at every step.
-   Clearly distinguish a simulated visual explanation from actual
    measured model values. Prefer actual values for all data-driven
    animation.
-   Include numerical tests for forward-pass shapes, finite loss,
    training progress, and deterministic behavior where applicable.

### F. Model import and real-model explorer

-   Start with ONNX as the first imported format.
-   Validate file extension, content, file size, parser errors, and
    model metadata.
-   Parse graph nodes, operators, edges, initializers, tensor names, and
    known shapes/types.
-   Report unknown or unsupported operators explicitly; do not silently
    discard graph information.
-   Keep imported model inspection separate from execution. Add ONNX
    Runtime inference only when implemented and tested.
-   Expose tensor summaries and bounded slices, with strict size limits.
-   Make model capabilities explicit in the UI.
-   Do not execute uploaded models automatically on upload.
-   Never unpickle untrusted Python objects or load untrusted PyTorch
    checkpoints using unsafe deserialization.
-   Treat imported files as untrusted input and isolate model execution
    when it is introduced.

### G. API and events

Use versioned endpoints under `/api/v1`. Initial endpoints: -
`GET /api/v1/health` - `POST /api/v1/models/educational` -
`GET /api/v1/models/{model_id}/graph` -
`POST /api/v1/models/{model_id}/forward` -
`POST /api/v1/models/{model_id}/training` -
`GET /api/v1/models/{model_id}/metrics` - `POST /api/v1/import/onnx` -
`GET /api/v1/models/{model_id}/tensors/{tensor_id}`

Training event channel: - `WS /api/v1/training/{job_id}/events`

Use Pydantic request/response models, clear HTTP status codes, request
validation, and consistent error responses. Document endpoint behavior
in `docs/api.md` when the API grows. Define event types and payloads in
a shared schema.

### H. Persistence

-   Begin with in-memory or local development storage if needed, but
    keep storage behind service/repository interfaces.
-   Save project definitions and user preferences before attempting
    complex experiment persistence.
-   Add PostgreSQL only when persistent multi-user projects are
    implemented.
-   If object storage is introduced for model files, document the
    provider and access controls.
-   Never commit uploaded models, datasets, secrets, local databases,
    `.env`, or generated artifacts.

### I. Testing and quality

-   Backend unit and API tests using Pytest and HTTPX.
-   Frontend component tests for important controls, state
    synchronization, error states, and view switching.
-   Integration tests for create model → graph response → forward pass.
-   Tests for malformed uploads, oversized tensor requests, unsupported
    operators, and invalid architecture configurations.
-   TypeScript strict mode; Python type hints for public interfaces.
-   Ruff for Python linting/formatting.
-   ESLint and formatter for frontend code.
-   Add a README section with exact commands that have actually been
    verified.
-   Do not claim a test passed unless it was run and passed.

### J. Security and operational requirements

-   Keep secrets in environment variables; provide `.env.example` with
    placeholders only.
-   Configure CORS through environment settings, not wildcard origins in
    production.
-   Enforce upload limits and bounded tensor requests.
-   Avoid blocking the FastAPI event loop with long training jobs.
-   Isolate long-running jobs and handle cancellation.
-   Add structured logs and safe error messages without exposing secrets
    or file paths.
-   Do not expose local filesystem paths or arbitrary file reads through
    API parameters.
-   Provide Docker support after the local development workflow is
    stable.
-   Add health checks and graceful error handling.
-   Document supported formats, limitations, and expected resource
    requirements.

## Development sequence

Implement in this order, committing a working state after each
milestone:

1.  **Foundation**
    -   Inspect repo, create scaffold, configure frontend/backend.
    -   Add health endpoint and connection indicator.
    -   Add shared schema and basic tests.
2.  **2D architecture**
    -   Build a feedforward network editor and graph renderer.
    -   Add inspector and selection state.
3.  **3D architecture**
    -   Render the same graph in 3D.
    -   Synchronize selection and camera controls.
4.  **Real educational computation**
    -   Add PyTorch model creation, forward pass, activations, training,
        metrics, and gradient capture.
    -   Add WebSocket streaming and cancellation.
5.  **ONNX inspection**
    -   Add safe upload and graph normalization.
    -   Add tensor summaries/slices and capability reporting.
6.  **Advanced visualizations**
    -   Add CNN feature maps and transformer attention only after core
        architecture and tests are stable.
7.  **Persistence and deployment**
    -   Add saved projects, database only when needed, Docker,
        deployment documentation, and production security review.

## Required workflow for the coding agent

For each milestone: 1. Inspect the current repository and read its
README/docs. 2. Summarize the next small implementation step and
relevant assumptions. 3. Implement the smallest coherent vertical slice.
4. Run relevant tests, type checks, lint, and build commands. 5. Fix
failures caused by the change. 6. Update documentation and examples. 7.
Report exactly what changed, commands run, results, known limitations,
and the next step. 8. Never claim a feature is complete when it is only
a placeholder. 9. Do not delete user work or overwrite existing
configuration without inspecting it first. 10. Avoid unrelated refactors
and unnecessary dependencies.

## Definition of done

A milestone is done only when: - The feature is connected end-to-end,
not just represented by static mock UI. - Relevant tests pass. -
Loading, error, and empty states exist. - Data is validated at API
boundaries. - The documentation reflects actual behavior. - Limitations
are explicit. - The app can be started using documented commands.

## First task

Start with Milestone 1 only. Inspect the repository, scaffold the
project if necessary, create the React + TypeScript + Vite frontend and
FastAPI backend, implement `GET /api/v1/health`, add a small dashboard
shell that checks backend health, configure CORS for local development,
add a basic test for the health endpoint, and document verified startup
commands. Do not implement the entire roadmap in the first change.
