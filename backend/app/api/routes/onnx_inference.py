from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request

from app.api.routes.models import Repository, _stored
from app.core.errors import ApiError
from app.schemas.errors import ErrorResponse
from app.schemas.inference import RunRequest, RunResponse
from app.services.model_repository import ImportedModelRecord
from app.services.onnx_runtime import OnnxRuntimeService, execute

router = APIRouter(prefix="/models", tags=["inference"])

ModelId = Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]


def get_runtime(request: Request) -> OnnxRuntimeService:
    runtime: OnnxRuntimeService = request.app.state.onnx_runtime
    return runtime


Runtime = Annotated[OnnxRuntimeService, Depends(get_runtime)]


# A plain `def`: waiting on the worker happens in the thread pool.
@router.post(
    "/{model_id}/run",
    response_model=RunResponse,
    responses={
        code: {"model": ErrorResponse} for code in (400, 404, 422, 429, 500, 504)
    },
)
def run_imported_model(
    model_id: ModelId, body: RunRequest, repository: Repository, runtime: Runtime
) -> RunResponse:
    """Run an imported ONNX model on validated inputs, in an isolated worker.

    Returns the graph outputs and the outputs of each operator (all of them
    for graphs up to 200 operators, else those listed in `capture`), each as
    a bounded summary and slice.
    """
    record = _stored(repository, model_id)
    if not isinstance(record, ImportedModelRecord):
        raise ApiError(
            400,
            "capability_unavailable",
            "This is an educational model; run it with POST /models/{model_id}/forward.",
        )
    return execute(runtime, record, body)
