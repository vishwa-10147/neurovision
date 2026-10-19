from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Path, Query, Request, status

from app.core.errors import not_found
from app.ml.educational_network import EducationalNetwork
from app.schemas.architecture import DenseNetworkConfig, EducationalModelCreated
from app.schemas.errors import ErrorResponse
from app.schemas.graph import ModelGraph
from app.schemas.training import (
    MAX_TENSOR_VALUES,
    ForwardRequest,
    ForwardResponse,
    MetricsResponse,
    TensorResponse,
    TrainingConfig,
    TrainingJobInfo,
)
from app.services.educational_graph import build_dense_graph
from app.services.inference import read_tensor, run_forward
from app.services.model_repository import ModelRecord, ModelRepository
from app.services.training_jobs import TrainingJobManager

router = APIRouter(prefix="/models", tags=["models"])

ModelId = Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
TensorId = Annotated[str, Path(min_length=1, max_length=64)]
ERRORS = {404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


def get_repository(request: Request) -> ModelRepository:
    repository: ModelRepository = request.app.state.model_repository
    return repository


def get_jobs(request: Request) -> TrainingJobManager:
    jobs: TrainingJobManager = request.app.state.training_jobs
    return jobs


Repository = Annotated[ModelRepository, Depends(get_repository)]
Jobs = Annotated[TrainingJobManager, Depends(get_jobs)]


def _record(repository: ModelRepository, model_id: str) -> ModelRecord:
    record = repository.get(model_id)
    if record is None:
        raise not_found(
            f"Model '{model_id}' was not found. It may have expired; create it again."
        )
    return record


@router.post(
    "/educational",
    status_code=status.HTTP_201_CREATED,
    response_model=EducationalModelCreated,
    responses={422: {"model": ErrorResponse}},
)
def create_educational_model(
    config: DenseNetworkConfig, repository: Repository
) -> EducationalModelCreated:
    model_id = f"edu_{uuid4().hex[:12]}"
    graph = build_dense_graph(config, model_id)
    repository.add(
        ModelRecord(graph=graph, config=config, network=EducationalNetwork(config))
    )
    return EducationalModelCreated(model_id=model_id, graph=graph)


@router.get("/{model_id}/graph", response_model=ModelGraph, responses=ERRORS)
def get_model_graph(model_id: ModelId, repository: Repository) -> ModelGraph:
    return _record(repository, model_id).graph


# Plain `def` endpoints run in FastAPI's thread pool, off the event loop.
@router.post("/{model_id}/forward", response_model=ForwardResponse, responses=ERRORS)
def forward(
    model_id: ModelId, body: ForwardRequest, repository: Repository
) -> ForwardResponse:
    return run_forward(_record(repository, model_id), body)


@router.get(
    "/{model_id}/tensors/{tensor_id}", response_model=TensorResponse, responses=ERRORS
)
def get_tensor(
    model_id: ModelId,
    tensor_id: TensorId,
    repository: Repository,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_TENSOR_VALUES)] = MAX_TENSOR_VALUES,
) -> TensorResponse:
    return read_tensor(_record(repository, model_id), tensor_id, offset, limit)


@router.post(
    "/{model_id}/training",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TrainingJobInfo,
    responses={**ERRORS, 409: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
def start_training(
    model_id: ModelId, config: TrainingConfig, repository: Repository, jobs: Jobs
) -> TrainingJobInfo:
    return jobs.start(_record(repository, model_id), config).info()


@router.get("/{model_id}/metrics", response_model=MetricsResponse, responses=ERRORS)
def get_metrics(
    model_id: ModelId, repository: Repository, jobs: Jobs
) -> MetricsResponse:
    record = _record(repository, model_id)
    job = jobs.get(record.last_job_id) if record.last_job_id else None
    return MetricsResponse(
        model_id=model_id,
        job=job.info() if job else None,
        history=list(record.metrics),
    )
