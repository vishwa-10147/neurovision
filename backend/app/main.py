from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import (
    classical,
    health,
    imports,
    lab,
    models,
    onnx_inference,
    projects,
    runs,
    training,
)
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.services.external_runs import ExternalRunManager
from app.services.lab import LabManager
from app.services.model_repository import InMemoryModelRepository
from app.services.onnx_runtime import OnnxRuntimeService
from app.services.project_store import ProjectStore
from app.services.training_jobs import TrainingJobManager

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    settings = get_settings()
    jobs = TrainingJobManager(max_active=settings.max_training_jobs)
    # Imported models run in a separate worker process, started on first use.
    onnx_runtime = OnnxRuntimeService(
        timeout_s=settings.inference_timeout_seconds,
        max_memory_bytes=settings.inference_max_memory_bytes,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # Cancel running jobs and wait for their threads before exiting.
        jobs.shutdown()
        onnx_runtime.close()
        app.state.lab.shutdown()

    app = FastAPI(title="NeuroVision API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    register_error_handlers(app)
    app.state.training_jobs = jobs
    app.state.onnx_runtime = onnx_runtime
    app.state.external_runs = ExternalRunManager(max_active=settings.max_external_runs)
    app.state.lab = LabManager(
        python=settings.lab_python,
        max_jobs=settings.lab_max_jobs,
        enabled=settings.lab_enabled
        if settings.lab_enabled is not None
        else settings.environment == "development",
    )
    app.state.project_store = ProjectStore(settings.projects_dir)
    # Evicting a model also stops any training that would keep updating it.
    app.state.model_repository = InMemoryModelRepository(
        on_evict=lambda record: jobs.cancel_for_model(record.model_id)
    )
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(models.router, prefix=API_PREFIX)
    app.include_router(training.router, prefix=API_PREFIX)
    app.include_router(imports.router, prefix=API_PREFIX)
    app.include_router(onnx_inference.router, prefix=API_PREFIX)
    app.include_router(runs.router, prefix=API_PREFIX)
    app.include_router(classical.router, prefix=API_PREFIX)
    app.include_router(lab.router, prefix=API_PREFIX)
    app.include_router(projects.router, prefix=API_PREFIX)
    return app


app = create_app()
