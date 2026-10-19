from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from app.schemas.architecture import DenseNetworkConfig, EducationalModelCreated
from app.schemas.errors import ErrorResponse
from app.schemas.graph import ModelGraph
from app.services.educational_graph import build_dense_graph
from app.services.model_repository import ModelRepository

router = APIRouter(prefix="/models", tags=["models"])

ModelId = Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]


def get_repository(request: Request) -> ModelRepository:
    repository: ModelRepository = request.app.state.model_repository
    return repository


Repository = Annotated[ModelRepository, Depends(get_repository)]


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
    repository.add(graph)
    return EducationalModelCreated(model_id=model_id, graph=graph)


@router.get(
    "/{model_id}/graph",
    response_model=ModelGraph,
    responses={404: {"model": ErrorResponse}},
)
def get_model_graph(model_id: ModelId, repository: Repository) -> ModelGraph:
    graph = repository.get_graph(model_id)
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' was not found. It may have expired; "
            "create it again.",
        )
    return graph
