from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request

from app.api.routes.models import Repository, _stored
from app.core.errors import ApiError
from app.schemas.errors import ErrorResponse
from app.schemas.inference import (
    RunRequest,
    RunResponse,
    TensorGridResponse,
    TensorMapResponse,
    TensorViewRequest,
)
from app.services.model_repository import ImportedModelRecord, ModelRepository
from app.services.onnx_runtime import OnnxRuntimeService, execute, view_tensor

router = APIRouter(prefix="/models", tags=["inference"])

ModelId = Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (400, 404, 410, 422, 429, 500, 504)
}


def get_runtime(request: Request) -> OnnxRuntimeService:
    runtime: OnnxRuntimeService = request.app.state.onnx_runtime
    return runtime


Runtime = Annotated[OnnxRuntimeService, Depends(get_runtime)]
ViewQuery = Annotated[TensorViewRequest, Query()]


def _imported(repository: ModelRepository, model_id: str) -> ImportedModelRecord:
    record = _stored(repository, model_id)
    if not isinstance(record, ImportedModelRecord):
        raise ApiError(
            400,
            "capability_unavailable",
            "This is an educational model; run it with POST /models/{model_id}/forward.",
        )
    return record


# Plain `def`s: waiting on the worker happens in the thread pool.
@router.post("/{model_id}/run", response_model=RunResponse, responses=ERRORS)
def run_imported_model(
    model_id: ModelId, body: RunRequest, repository: Repository, runtime: Runtime
) -> RunResponse:
    """Run an imported ONNX model on validated inputs, in an isolated worker.

    Returns the graph outputs and the outputs of each operator (all of them
    for graphs up to 200 operators, else those listed in `capture`), each as
    a bounded summary and slice.
    """
    return execute(runtime, _imported(repository, model_id), body)


@router.get("/{model_id}/maps", response_model=TensorMapResponse, responses=ERRORS)
def tensor_map(
    model_id: ModelId, query: ViewQuery, repository: Repository, runtime: Runtime
) -> TensorMapResponse:
    """One 2D map of a tensor from a recent run: its last two dimensions at
    the `fixed` leading indices (e.g. one feature-map channel, or one
    attention head), block-averaged to at most 65,536 cells."""
    result = view_tensor(runtime, _imported(repository, model_id), "map", query)
    return TensorMapResponse.model_validate(result)


@router.get("/{model_id}/grids", response_model=TensorGridResponse, responses=ERRORS)
def tensor_grid(
    model_id: ModelId, query: ViewQuery, repository: Repository, runtime: Runtime
) -> TensorGridResponse:
    """Thumbnails of every map along the third-to-last dimension (all
    channels, or all heads), at most 256, each at most 24 × 24 cells."""
    result = view_tensor(runtime, _imported(repository, model_id), "grid", query)
    return TensorGridResponse.model_validate(result)
