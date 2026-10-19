from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Path, Query, Request, status
from fastapi.concurrency import run_in_threadpool

from app.api.routes.models import Repository
from app.core.config import get_settings
from app.core.errors import ApiError, invalid, not_found
from app.ml.onnx_samples import SAMPLES, build_sample
from app.schemas.errors import ErrorResponse
from app.schemas.imports import ImportResponse, SampleModel
from app.services.model_repository import ImportedModelRecord, ModelRepository
from app.services.onnx_importer import OnnxImportError, import_onnx

router = APIRouter(prefix="/import", tags=["import"])

ERRORS = {
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}
FileName = Annotated[str, Query(min_length=1, max_length=255)]


def _too_large(limit: int) -> ApiError:
    return ApiError(
        413,
        "payload_too_large",
        f"The file is larger than the {limit / 1024 / 1024:.0f} MB upload limit.",
    )


async def _read_body(request: Request, limit: int) -> bytes:
    """Read the raw upload, refusing it as soon as it exceeds `limit` bytes."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise _too_large(limit)
    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > limit:
            raise _too_large(limit)
        chunks.append(chunk)
    return b"".join(chunks)


async def _import(
    data: bytes, file_name: str, repository: ModelRepository
) -> ImportResponse:
    model_id = f"onnx_{uuid4().hex[:12]}"
    try:
        # Parsing and shape inference are CPU-bound; keep them off the event loop.
        result = await run_in_threadpool(import_onnx, data, model_id, file_name)
    except OnnxImportError as error:
        raise ApiError(422, "invalid_model", str(error)) from error
    repository.add(
        ImportedModelRecord(
            graph=result.graph,
            initializers=result.initializers,
            size_bytes=len(data),
        )
    )
    return ImportResponse(model_id=model_id, graph=result.graph, report=result.report)


@router.post(
    "/onnx",
    status_code=status.HTTP_201_CREATED,
    response_model=ImportResponse,
    responses=ERRORS,
)
async def import_onnx_file(
    request: Request, repository: Repository, filename: FileName
) -> ImportResponse:
    """Import an ONNX file sent as the raw request body, for inspection only.

    The model is parsed in memory and never executed; external data
    references are not followed.
    """
    if not filename.lower().endswith(".onnx"):
        raise invalid("query.filename", "Only .onnx files can be imported.")
    data = await _read_body(request, get_settings().max_upload_bytes)
    if not data:
        raise invalid("body", "The file is empty.")
    return await _import(data, filename, repository)


@router.get("/onnx/samples", response_model=list[SampleModel])
def list_samples() -> list[SampleModel]:
    return SAMPLES


@router.post(
    "/onnx/samples/{name}",
    status_code=status.HTTP_201_CREATED,
    response_model=ImportResponse,
    responses=ERRORS,
)
async def import_sample(
    name: Annotated[str, Path(max_length=32)], repository: Repository
) -> ImportResponse:
    model = build_sample(name)
    if model is None:
        raise not_found(f"There is no sample named '{name}'.")
    return await _import(model.SerializeToString(), f"{name}.onnx", repository)
