# NeuroVision --- Skills and Engineering Standards

## Purpose

This document defines the skills, tools, and practices required to build
NeuroVision. It is also a learning checklist for contributors.

## 1. Required skill areas

### Frontend engineering

-   [ ] JavaScript and TypeScript fundamentals.
-   [ ] React components, hooks, composition, and controlled inputs.
-   [ ] Vite development and production builds.
-   [ ] Type-safe API clients and runtime validation.
-   [ ] State management with Zustand.
-   [ ] Responsive layouts and accessible interactions.
-   [ ] Testing UI behavior and error states.

### Graph and data visualization

-   [ ] Graph theory: nodes, edges, directed graphs, DAGs, and cycles.
-   [ ] Graph layout and stable node identifiers.
-   [ ] SVG and Canvas rendering trade-offs.
-   [ ] React Flow for high-level architecture editing.
-   [ ] Three.js coordinate systems, cameras, materials, lighting, and
    raycasting.
-   [ ] React Three Fiber scene composition.
-   [ ] Instanced rendering, batching, filtering, and levels of detail.
-   [ ] Animation that reflects actual data rather than fabricated
    behavior.

### Python backend

-   [ ] Python 3.12, type hints, virtual environments, and package
    management.
-   [ ] FastAPI routing, dependency injection, middleware, and error
    handling.
-   [ ] Pydantic request and response validation.
-   [ ] Async I/O and avoiding blocking operations in the event loop.
-   [ ] WebSocket lifecycle and reconnection handling.
-   [ ] Pytest and HTTPX API testing.
-   [ ] Ruff linting and formatting.

### Machine learning

-   [ ] Tensors, shapes, dimensions, and broadcasting.
-   [ ] Dense layers and parameter counts.
-   [ ] Activation functions: ReLU, sigmoid, tanh.
-   [ ] Forward propagation and loss functions.
-   [ ] Gradient descent, backpropagation, and autograd.
-   [ ] Training, validation, and test splits.
-   [ ] Overfitting, generalization, and common metrics.
-   [ ] PyTorch modules, hooks, optimizers, and training loops.
-   [ ] CPU/GPU execution and memory limits.
-   [ ] Distinguishing model internals from explanatory approximations.

### Model interoperability

-   [ ] ONNX graph, nodes, edges, operators, initializers, and tensor
    metadata.
-   [ ] Static and dynamic tensor shapes.
-   [ ] Operator compatibility and unknown-operator handling.
-   [ ] ONNX Runtime input/output conventions.
-   [ ] Safe model import and execution boundaries.
-   [ ] Tensor slicing, summaries, and memory-aware inspection.

### Software engineering

-   [ ] Git branches, focused commits, pull requests, and code review.
-   [ ] Modular architecture and dependency boundaries.
-   [ ] API contract design and schema versioning.
-   [ ] Unit, integration, and regression tests.
-   [ ] Logging, configuration, and environment variables.
-   [ ] Reproducible dependencies and documented setup.
-   [ ] Docker and deployment fundamentals.
-   [ ] Profiling and performance measurement.

### Security and reliability

-   [ ] File upload validation and size limits.
-   [ ] Safe handling of untrusted model files.
-   [ ] Avoiding unsafe pickle deserialization.
-   [ ] Request and resource limits.
-   [ ] Secrets management and CORS configuration.
-   [ ] Job cancellation and cleanup.
-   [ ] Safe error messages and privacy-aware logs.
-   [ ] Dependency updates and vulnerability review.

## 2. Tooling checklist

### Required for early development

-   [ ] Git
-   [ ] Node.js LTS and npm
-   [ ] Python 3.12
-   [ ] VS Code or another code editor
-   [ ] React, TypeScript, Vite
-   [ ] FastAPI, Pydantic, Uvicorn
-   [ ] PyTorch, NumPy, ONNX
-   [ ] Pytest, HTTPX, Ruff
-   [ ] Browser developer tools

### Required as features are introduced

-   [ ] React Three Fiber and Three.js for 3D.
-   [ ] React Flow for high-level architecture editing.
-   [ ] Zustand for shared client state.
-   [ ] Recharts for metrics.
-   [ ] ONNX Runtime for compatible imported-model inference.
-   [ ] Docker for deployment.
-   [ ] PostgreSQL only when durable multi-user persistence is required.

Do not install every optional tool before it is needed. Keep
dependencies aligned with implemented features.

## 3. Coding standards

### TypeScript

-   Enable strict type checking.
-   Avoid `any` unless there is a documented reason.
-   Validate external API data at runtime.
-   Keep semantic graph data separate from renderer objects.
-   Use stable IDs for nodes and edges.
-   Keep components focused and testable.

### Python

-   Add type hints to public interfaces.
-   Validate API inputs with Pydantic.
-   Separate routes, services, schemas, and ML logic.
-   Keep long-running training out of request handlers.
-   Avoid global mutable state for model jobs.
-   Return safe errors and log useful context.

### ML correctness

-   Use actual activations and metrics when claiming to show model
    behavior.
-   Test tensor dimensions and numerical stability.
-   Seed sample experiments when reproducibility matters.
-   Label normalized or sampled values clearly.
-   Do not imply every imported model can be trained or has gradients
    available.

### Visualization

-   Include legends for color, width, and animation semantics.
-   Keep selection consistent between 2D, 3D, and the inspector.
-   Provide a detail limit for large graphs.
-   Keep animation optional or reduced for accessibility and
    performance.
-   Do not render huge tensors or billions of parameters directly.

## 4. Definition of done

A feature is complete only when: - \[ \] Its intended behavior works
end-to-end. - \[ \] Inputs and outputs are validated. - \[ \] Loading,
empty, and error states are implemented where applicable. - \[ \]
Relevant automated tests pass. - \[ \] Linting, type checks, and builds
have been run where configured. - \[ \] Documentation is updated. - \[
\] Limitations are stated honestly. - \[ \] No secrets, uploaded models,
or generated artifacts were accidentally committed.

## 5. Recommended learning sequence

1.  React + TypeScript + Vite fundamentals.
2.  Graph data structures and a simple 2D renderer.
3.  Three.js scene basics and camera controls.
4.  Python + FastAPI and typed API contracts.
5.  NumPy tensors and a basic feedforward network.
6.  PyTorch forward pass, training loop, and autograd.
7.  WebSocket event streaming.
8.  ONNX graph parsing and tensor metadata.
9.  Rendering performance, profiling, and resource limits.
10. Docker, deployment, persistence, and security hardening.

## 6. Quality commands

Use commands configured by the project. The following are typical
examples after scripts and dependencies exist.

Frontend:

``` powershell
npm run lint
npm run build
npm run test
```

Backend:

``` powershell
pytest
ruff check .
ruff format --check .
```

Do not document a command as supported until the relevant script or tool
is configured. Do not claim success without running it.

## 7. Coding-agent collaboration rules

When using an AI coding agent: 1. Read `PROMPT.md`, `ARCHITECTURE.md`,
and `PLAN.md` before implementation. 2. Work on one milestone at a time.
3. Inspect existing files before editing. 4. Do not replace user work
without a clear reason. 5. Implement functional code rather than static
mockups. 6. Run tests and report actual results. 7. Update docs when
behavior changes. 8. Separate implemented features from planned
features. 9. Ask before making irreversible changes or adding
unnecessary infrastructure. 10. End each work session with changed
files, verification results, limitations, and the next recommended task.
