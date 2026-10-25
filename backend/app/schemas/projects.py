"""Versioned local project documents for architecture definitions."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.architecture import DenseNetworkConfig

PROJECT_SCHEMA_VERSION = 1
ProjectId = Annotated[
    str, Field(min_length=1, max_length=64, pattern=r"^proj_[A-Za-z0-9]+$")
]


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=80)]
    config: DenseNetworkConfig
    description: Annotated[str, Field(max_length=500)] = ""


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    description: str
    schema_version: int
    created_at: str
    updated_at: str


class ProjectDocument(ProjectSummary):
    config: DenseNetworkConfig


class ProjectList(BaseModel):
    projects: list[ProjectSummary]
