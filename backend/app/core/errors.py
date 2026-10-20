"""Exception handlers that return the shared error envelope.

Messages are safe for clients: no stack traces, file paths, or echoed input.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.errors import ErrorBody, ErrorDetail, ErrorResponse

logger = logging.getLogger("neurovision.api")

_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    429: "too_many_requests",
    422: "validation_error",
}


class ApiError(Exception):
    """An expected failure that services raise to return the error envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: list[ErrorDetail] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []


def not_found(message: str) -> ApiError:
    return ApiError(404, "not_found", message)


def invalid(field: str, message: str) -> ApiError:
    """A 422 for a rule Pydantic cannot check alone (e.g. depends on the model)."""
    return ApiError(
        422,
        "validation_error",
        "The request did not pass validation.",
        [ErrorDetail(field=field, message=message)],
    )


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, details=details or [])
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def _http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    if exc.status_code == 404 and message == "Not Found":
        message = f"No API route matches {request.method} {request.url.path}."
    return error_response(
        exc.status_code, _CODES.get(exc.status_code, "http_error"), message
    )


async def _validation_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        ErrorDetail(
            field=".".join(str(part) for part in err.get("loc", ())) or None,
            message=str(err.get("msg", "Invalid value.")),
        )
        for err in exc.errors()
    ]
    return error_response(
        422, "validation_error", "The request did not pass validation.", details
    )


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return error_response(500, "internal_error", "An unexpected server error occurred.")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_exception)
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(Exception, _unhandled_exception)
