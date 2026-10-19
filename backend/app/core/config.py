"""Environment-backed application settings."""

from functools import lru_cache
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEUROVISION_", env_file=".env", extra="ignore"
    )

    environment: str = "development"
    # Training jobs that may run at once; more are rejected with 429.
    max_training_jobs: int = 2
    # Largest model file accepted for import.
    max_upload_bytes: int = 64 * 1024 * 1024
    # Comma-separated list in the environment, e.g.
    # NEUROVISION_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
