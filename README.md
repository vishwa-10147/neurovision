# NeuroVision

**NeuroVision** is an interactive neural-network visualization platform
for exploring network architecture, observing signal propagation,
learning how training changes model parameters, and inspecting supported
real trained models.

> Status: Early development. Implemented so far: a dashboard with a
> backend health check; the **Architecture Studio**, where you build a
> dense network (validated limits, ReLU/sigmoid/tanh) whose graph is
> built by the backend and rendered in 2D with pan, zoom, fit, and a
> synchronized inspector; shared graph, event, and error contracts. No
> PyTorch model exists yet, so there are no forward passes, training,
> activations, or weights. The 3D view, Training Lab, Model Explorer,
> and Learning Mode are shown in the UI as "planned". Features described as planned are not
> considered implemented until their code and tests exist.

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

The commands below were verified on Windows 11 with Node.js 24 and
Python 3.12.10. Only the dependencies the implemented features need are
installed; PyTorch, ONNX, Three.js, React Flow, Zustand, and Recharts
are added in the milestones that use them.

### 1. Backend

From the repository root:

``` powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements.txt` holds pinned runtime dependencies;
`requirements-dev.txt` adds the test and lint tools.

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

### PyTorch (from Milestone 4)

Install PyTorch using the official selector at
https://pytorch.org/get-started/locally/ and choose the appropriate
platform and compute option. Do not assume that a locally installed CUDA
toolkit automatically determines the correct PyTorch package.

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

Request and response details, limits, and the in-memory storage
limitation are in [`docs/api.md`](docs/api.md). Rendering decisions and
the large-graph threshold are in
[`docs/decisions/0001-2d-renderer.md`](docs/decisions/0001-2d-renderer.md).

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
