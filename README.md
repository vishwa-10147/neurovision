# NeuroVision

**NeuroVision** is an interactive neural-network visualization platform
for exploring network architecture, observing signal propagation,
learning how training changes model parameters, and inspecting supported
real trained models.

> Status: Early development. Implemented so far:
>
> -   **Architecture Studio**: build a dense network (validated limits,
>     ReLU/sigmoid/tanh), backed by a seeded PyTorch model and shown in
>     switchable 2D and 3D views with a synchronized inspector. Run a
>     forward pass to shade units by the values the model computes; the
>     inspector shows real activations, weights, biases, and gradients.
> -   **Training Lab**: train on deterministic moons, circles, or spiral
>     data in a background job, with live loss and accuracy charts and
>     streamed activations over a WebSocket, plus cancellation.
> -   **Model Explorer**: import a trained ONNX model (e.g. exported from
>     your own PyTorch project) or a built-in sample, and inspect its
>     operator graph in 2D and 3D, its shapes and attributes, and its
>     weights. Files are parsed safely for inspection and never run. See
>     [docs/importing-onnx.md](docs/importing-onnx.md).
> -   A dashboard with a backend health check, and shared graph, event,
>     and error contracts.
>
> Learning Mode is shown in the UI as "planned", and running imported
> models arrives in Milestone 6. The 3D view needs WebGL. Features
> described as planned are not considered implemented until their code
> and tests exist.

## Vision

NeuroVision brings together two complementary experiences:

1.  **Learning Lab:** Create small neural networks, run forward passes,
    train them, and inspect real activations, weights, gradients, and
    metrics.
2.  **Model Explorer:** Import supported models, starting with ONNX, and
    explore their graph structure, operators, tensor shapes, metadata,
    and selected tensor values.

Both experiences use a shared normalized graph model so that the same
model can be viewed in 2D and 3D.

## Planned features

-   2D architecture editor and graph renderer.
-   3D neural-network explorer with camera controls.
-   Synchronized node/edge selection and inspector.
-   Educational feedforward networks using PyTorch.
-   Real activation and gradient capture.
-   Training metrics and WebSocket event streaming.
-   ONNX graph import and inspection.
-   Bounded tensor summaries and slices.
-   Learning guides for dense networks, CNNs, and transformers.
-   Saved projects and deployment support in later milestones.

## Technology stack

### Frontend

-   React, TypeScript, Vite
-   React Three Fiber, Three.js
-   React Flow (`@xyflow/react`)
-   Zustand
-   Recharts
-   Zod

### Backend

-   Python 3.12
-   FastAPI, Pydantic, Uvicorn
-   PyTorch, NumPy
-   ONNX; ONNX Runtime when inference support is implemented
-   WebSockets
-   Pytest, HTTPX, Ruff

## Repository layout

``` text
neurovision/
├── frontend/             # React + TypeScript application
├── backend/              # FastAPI and ML services
├── shared/schemas/       # Graph and event contracts
├── docs/                 # Design and API documentation
├── tests/                # Cross-component tests
├── PROMPT.md             # Master implementation prompt for coding agents
├── ARCHITECTURE.md       # System design and boundaries
├── PLAN.md               # Milestones and acceptance criteria
└── SKILLS.md             # Required tools and engineering practices
```

Create files progressively; this tree describes the target organization,
not a claim that every module already exists.

## Prerequisites

-   Git
-   Node.js LTS and npm
-   Python 3.12
-   VS Code or another editor
-   Optional NVIDIA GPU and compatible PyTorch CUDA runtime for
    accelerated training

Verify the installations:

``` powershell
node --version
npm --version
python --version
git --version
```

## Setup on Windows PowerShell

The commands below were verified on Windows 11 with Node.js 24,
Python 3.12.10, the CPU build of PyTorch 2.14.1, and onnx 1.23.2. Only
the dependencies the implemented features need are installed; ONNX
Runtime and React Flow are added in the milestones that use them.

### 1. Backend

From the repository root, install PyTorch first for your platform, then
the pinned requirements:

``` powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements-dev.txt
```

The CPU build is enough for the educational networks (each training run
takes seconds). For a GPU build, pick the matching command at
https://pytorch.org/get-started/locally/ instead; do not assume a
locally installed CUDA toolkit determines the right package.
`requirements.txt` pins `torch==2.14.1`, which local builds such as
`2.14.1+cpu` satisfy. `requirements-dev.txt` adds the test and lint
tools.

### 2. Frontend

From the repository root:

``` powershell
cd frontend
npm install
```

### 3. Start the services

Terminal A (backend):

``` powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

Terminal B (frontend):

``` powershell
cd frontend
npm run dev
```

Local URLs:

-   Frontend: http://localhost:5173
-   Health endpoint: http://localhost:8000/api/v1/health
-   FastAPI interactive API docs: http://localhost:8000/docs

The dashboard shows "Backend online" with the service version when the
health check succeeds, and an error with startup instructions when it
fails or does not answer within 5 seconds. It re-checks every 15
seconds.

### Using other ports

If 8000 or 5173 is taken, point the two services at each other through
environment variables (see `backend/.env.example` and
`frontend/.env.example`). For example, backend on 8001 and frontend on
5174:

``` powershell
# backend/
$env:NEUROVISION_CORS_ORIGINS = "http://localhost:5174"
uvicorn app.main:app --reload --port 8001
```

``` powershell
# frontend/
$env:VITE_API_BASE_URL = "http://localhost:8001"
npm run dev -- --port 5174
```

### 4. Quality checks

Backend (from `backend/` with the virtual environment active):

``` powershell
pytest
ruff check .
ruff format --check .
```

Frontend (from `frontend/`):

``` powershell
npm run lint        # oxlint
npm run typecheck   # tsc -b, strict mode
npm test            # Vitest + Testing Library
npm run build
```

## PyTorch verification

After installing PyTorch, run:

``` powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CPU execution is sufficient for early educational examples. GPU support
can be enabled when the installed PyTorch build and hardware runtime are
compatible.

## Implemented API

| Endpoint                                | Purpose                                         |
|-----------------------------------------|-------------------------------------------------|
| `GET /api/v1/health`                    | Service health                                  |
| `POST /api/v1/models/educational`       | Validate a dense-network config, build its graph |
| `GET /api/v1/models/{model_id}/graph`   | Fetch a created model's normalized graph        |
| `POST /api/v1/models/{model_id}/forward` | Forward pass with every layer's real values    |
| `GET /api/v1/models/{model_id}/tensors/{tensor_id}` | Bounded weights, biases, gradients  |
| `POST /api/v1/models/{model_id}/training` | Start a background training job               |
| `GET /api/v1/models/{model_id}/metrics` | Latest job and per-epoch metrics                |
| `GET /api/v1/training/{job_id}`         | Job status                                      |
| `POST /api/v1/training/{job_id}/cancel` | Cancel a job                                    |
| `WS /api/v1/training/{job_id}/events`   | Replayed and live training events               |
| `POST /api/v1/import/onnx?filename=…`   | Import an ONNX file (raw body) for inspection   |
| `GET /api/v1/import/onnx/samples`       | List built-in sample models                     |
| `POST /api/v1/import/onnx/samples/{name}` | Import a sample model                         |

Request and response details, limits, and the in-memory storage
limitation are in [`docs/api.md`](docs/api.md). Design decisions are in
[`docs/decisions/`](docs/decisions/): the
[2D renderer](docs/decisions/0001-2d-renderer.md), the
[3D renderer](docs/decisions/0002-3d-renderer.md),
[training jobs and event streaming](docs/decisions/0003-training-jobs.md),
and [ONNX import](docs/decisions/0004-onnx-import.md).

Errors from any endpoint use a shared envelope,
`{"error": {"code", "message", "details"}}`. The model graph, training
event, error, and health contracts, and how to keep the Pydantic, JSON
Schema, and Zod versions in sync, are documented in
[`docs/schemas.md`](docs/schemas.md). CORS origins come from
`NEUROVISION_CORS_ORIGINS`; the default allows only
`http://localhost:5173` and `http://127.0.0.1:5173`.

## API conventions

-   Use `/api/v1` for versioned REST endpoints.
-   Validate requests and responses with Pydantic.
-   Use WebSockets for compact training events.
-   Keep large tensors outside graph JSON and serve bounded summaries or
    slices.
-   Return clear error responses for unsupported operators, invalid
    architecture settings, malformed files, and oversized requests.

Planned endpoints are documented in `ARCHITECTURE.md`.

## Security notes

-   Treat model uploads as untrusted.
-   Enforce upload and tensor-request size limits.
-   Never load untrusted pickle files or unsafe serialized Python model
    objects.
-   Do not execute imported models automatically on upload.
-   Keep secrets out of source control.
-   Configure CORS explicitly for each environment.
-   Isolate inference and training jobs when execution of imported
    models is introduced.

## Development principles

1.  Build one working vertical slice at a time.
2.  Use a shared graph schema for all visualization modes.
3.  Distinguish actual measured values from explanatory animation.
4.  Keep semantic graph data separate from visual layout coordinates.
5.  Test numerical correctness, API validation, and renderer
    synchronization.
6.  Update documentation only to reflect behavior that exists.
7.  Do not claim tests or features are complete without verifying them.

## Roadmap

See [`PLAN.md`](PLAN.md) for the milestone sequence and acceptance
criteria, and [`ARCHITECTURE.md`](ARCHITECTURE.md) for system
boundaries.

## Contributing

-   Create a focused branch for each feature.
-   Keep pull requests small and testable.
-   Include tests for bug fixes and numerical logic.
-   Run formatting, linting, type checks, tests, and production builds
    before merging.
-   Document limitations and any intentional trade-offs.

## License

Choose and add a license before public distribution. Do not imply that
the project has a license until the corresponding license file is
present.
