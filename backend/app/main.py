from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import health, models
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.services.model_repository import InMemoryModelRepository

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="NeuroVision API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    register_error_handlers(app)
    app.state.model_repository = InMemoryModelRepository()
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(models.router, prefix=API_PREFIX)
    return app


app = create_app()
