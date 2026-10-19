from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import health, imports, models, training
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.services.model_repository import InMemoryModelRepository
from app.services.training_jobs import TrainingJobManager

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    settings = get_settings()
    jobs = TrainingJobManager(max_active=settings.max_training_jobs)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # Cancel running jobs and wait for their threads before exiting.
        jobs.shutdown()

    app = FastAPI(title="NeuroVision API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    register_error_handlers(app)
    app.state.training_jobs = jobs
    # Evicting a model also stops any training that would keep updating it.
    app.state.model_repository = InMemoryModelRepository(
        on_evict=lambda record: jobs.cancel_for_model(record.model_id)
    )
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(models.router, prefix=API_PREFIX)
    app.include_router(training.router, prefix=API_PREFIX)
    app.include_router(imports.router, prefix=API_PREFIX)
    return app


app = create_app()
