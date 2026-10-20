"""Responses for imported models."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.graph import ModelGraph
from app.schemas.inference import InputSpec


class UnsupportedOperator(BaseModel):
    op_type: str
    domain: str
    count: int


class OpsetImport(BaseModel):
    domain: str
    version: int


class ImportReport(BaseModel):
    format: Literal["onnx"] = "onnx"
    file_name: str
    file_bytes: int
    operator_count: int
    edge_count: int
    initializer_count: int
    # Elements across initializers (learned weights and constants).
    parameter_count: int
    # Initializers whose data lives in files that were not uploaded.
    external_tensor_count: int
    unsupported_operators: list[UnsupportedOperator] = Field(default_factory=list)
    opsets: list[OpsetImport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Inputs a run must provide, as declared in the file.
    inputs: list[InputSpec] = Field(default_factory=list)
    # Why the model cannot be run here; empty when it is expected to run.
    inference_blockers: list[str] = Field(default_factory=list)


class ImportResponse(BaseModel):
    model_id: str
    graph: ModelGraph
    report: ImportReport


class SampleModel(BaseModel):
    name: str
    title: str
    description: str
