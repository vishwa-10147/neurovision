"""Bounded 2D views of tensors: one detailed map, or a grid of thumbnails.

Feature maps ([N, C, H, W]) and attention weights ([batch, heads, query,
key]) are both stacks of 2D maps. These helpers pick one map by fixing the
leading indices, or one map per index along an axis, and shrink each map
by block averaging when it exceeds a cell budget. Pure NumPy, so they can
be tested without ONNX Runtime and used inside the inference worker.
"""

import math
from typing import Any

import numpy as np


class MapRequestError(ValueError):
    """The request does not fit the tensor; the message is client-safe."""


def _check_fixed(shape: tuple[int, ...], fixed: list[int], free_dims: int) -> None:
    if len(shape) < free_dims:
        raise MapRequestError(
            f"The tensor has {len(shape)} dimensions; this view needs at least {free_dims}."
        )
    expected = len(shape) - free_dims
    if len(fixed) != expected:
        raise MapRequestError(
            f"Give {expected} leading index(es) for shape {list(shape)}, got {len(fixed)}."
        )
    for axis, (index, size) in enumerate(zip(fixed, shape, strict=False)):
        if not 0 <= index < size:
            raise MapRequestError(
                f"Index {index} is out of range for dimension {axis} (size {size})."
            )


def downsample(matrix: np.ndarray, max_cells: int) -> tuple[np.ndarray, int]:
    """Block-average a 2D array until it has at most `max_cells` cells.

    Returns the (possibly) smaller array and the block size used (1 = exact).
    Non-finite values are ignored within a block; an all-non-finite block is NaN.
    """
    rows, cols = matrix.shape
    if rows * cols <= max_cells:
        return matrix.astype(np.float64), 1
    stride = math.ceil(math.sqrt(rows * cols / max_cells))
    while math.ceil(rows / stride) * math.ceil(cols / stride) > max_cells:
        stride += 1
    padded_rows, padded_cols = (
        math.ceil(rows / stride) * stride,
        math.ceil(cols / stride) * stride,
    )
    padded = np.full((padded_rows, padded_cols), np.nan)
    finite = np.where(np.isfinite(matrix), matrix, np.nan).astype(np.float64)
    padded[:rows, :cols] = finite
    blocks = padded.reshape(
        padded_rows // stride, stride, padded_cols // stride, stride
    )
    with np.errstate(invalid="ignore"), _quiet_nanmean():
        return np.nanmean(blocks, axis=(1, 3)), stride


class _quiet_nanmean:
    """Silence NumPy's 'mean of empty slice' warning for all-NaN blocks."""

    def __enter__(self) -> None:
        import warnings

        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc: object) -> None:
        self._catch.__exit__(*exc)


def _cells(matrix: np.ndarray) -> list[list[float | None]]:
    return [[float(v) if np.isfinite(v) else None for v in row] for row in matrix]


def _stats(matrix: np.ndarray) -> dict[str, Any]:
    finite = matrix[np.isfinite(matrix)]
    return {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "non_finite_count": int(matrix.size - finite.size),
    }


def tensor_map(array: np.ndarray, fixed: list[int], max_cells: int) -> dict[str, Any]:
    """The 2D map in the last two dimensions, at `fixed` leading indices."""
    shape = tuple(int(d) for d in np.shape(array))
    _check_fixed(shape, fixed, 2)
    matrix = np.asarray(array, dtype=np.float64)[tuple(fixed)]
    small, stride = downsample(matrix, max_cells)
    return {
        "shape": list(shape),
        "fixed": fixed,
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "stride": stride,
        "values": _cells(small),
        **_stats(matrix),
    }


def tensor_grid(
    array: np.ndarray, fixed: list[int], max_items: int, thumb_cells: int
) -> dict[str, Any]:
    """One thumbnail per index along the third-to-last axis (channels, heads)."""
    shape = tuple(int(d) for d in np.shape(array))
    _check_fixed(shape, fixed, 3)
    stack = np.asarray(array, dtype=np.float64)[tuple(fixed)]
    items = []
    for index in range(min(stack.shape[0], max_items)):
        small, stride = downsample(stack[index], thumb_cells)
        items.append(
            {
                "index": index,
                "stride": stride,
                "values": _cells(small),
                **_stats(stack[index]),
            }
        )
    return {
        "shape": list(shape),
        "fixed": fixed,
        "axis": len(fixed),
        "items_total": int(stack.shape[0]),
        "rows": int(stack.shape[1]),
        "cols": int(stack.shape[2]),
        "items": items,
        **_stats(stack[: len(items)]),
    }
