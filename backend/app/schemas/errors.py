"""Consistent error envelope returned by every API error response."""

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    # Dotted location of the offending field, e.g. "body.layers.0.width".
    field: str | None = None
    message: str


class ErrorBody(BaseModel):
    # Stable machine-readable code, e.g. "validation_error" or "not_found".
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody
