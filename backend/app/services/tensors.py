"""Bounded tensor summaries and slices, shared by every model source."""

import numpy as np

from app.core.errors import ApiError, invalid, not_found
from app.schemas.training import MAX_TENSOR_VALUES, TensorResponse, TensorSummary
from app.services.model_repository import ImportedModelRecord
from app.services.onnx_importer import is_external


def tensor_response(
    tensor_id: str,
    shape: list[int],
    dtype: str,
    flat: np.ndarray,
    offset: int,
    limit: int,
) -> TensorResponse:
    """Summary over the whole tensor plus a bounded row-major slice.

    Non-finite values (NaN, ±inf) cannot be sent as JSON numbers: they are
    counted, excluded from the summary, and returned as null in the slice.
    """
    count = int(flat.size)
    if offset > count:
        raise invalid(
            "query.offset",
            f"Offset {offset} is past the end of the tensor ({count} values).",
        )
    values = flat.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    finite_values = values[finite]
    summary = (
        TensorSummary(
            min=float(finite_values.min()),
            max=float(finite_values.max()),
            mean=float(finite_values.mean()),
            std=float(finite_values.std()),
        )
        if finite_values.size
        else None
    )
    window = values[offset : offset + min(limit, MAX_TENSOR_VALUES)]
    return TensorResponse(
        tensor_id=tensor_id,
        shape=shape,
        dtype=dtype,
        count=count,
        summary=summary,
        non_finite_count=int(count - finite_values.size),
        offset=offset,
        values=[float(v) if np.isfinite(v) else None for v in window],
        truncated=offset + window.size < count,
    )


def read_onnx_tensor(
    record: ImportedModelRecord, tensor_id: str, offset: int, limit: int
) -> TensorResponse:
    from onnx import TensorProto, numpy_helper

    tensor = record.initializers.get(tensor_id)
    if tensor is None:
        raise not_found(
            f"Tensor '{tensor_id}' does not exist in this model. Use ids like 't0'."
        )
    if is_external(tensor):
        raise not_found(
            "This tensor's values are stored in an external file that was not uploaded, so only "
            "its shape is available."
        )
    if tensor.data_type == TensorProto.STRING:
        raise ApiError(
            400, "unsupported_tensor", "String tensors cannot be summarized."
        )
    try:
        array = numpy_helper.to_array(tensor)
        flat = np.asarray(array, dtype=np.float64).reshape(-1)
    except Exception as error:
        raise ApiError(
            400, "unsupported_tensor", "This tensor's values could not be decoded."
        ) from error
    dtype = TensorProto.DataType.Name(tensor.data_type).lower()
    return tensor_response(tensor_id, list(tensor.dims), dtype, flat, offset, limit)
