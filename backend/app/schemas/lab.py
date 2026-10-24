"""Neural Lab: train built-in PyTorch models, configured in the browser.

The lab launches only NeuroVision's own demo scripts (sdk/examples), with
parameters validated against each template's bounds. The script streams to
Live Runs like any user script, so the graph, health, visuals and controls
work the same way.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ParamSpec(BaseModel):
    name: str
    label: str
    kind: Literal["int", "float", "choice"]
    default: float | str
    min: float | None = None
    max: float | None = None
    choices: list[str] | None = None
    help: str | None = None


class Template(BaseModel):
    id: str
    label: str
    family: str
    description: str
    # What the script draws in Live Runs besides charts and the graph.
    shows: list[str]
    params: list[ParamSpec]


class TemplateList(BaseModel):
    templates: list[Template]
    # False when the lab is turned off on this server (NEUROVISION_LAB_ENABLED).
    enabled: bool
    # The Python that runs the scripts, e.g. a CUDA build for the GPU.
    python: str | None


class LaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: Annotated[str, Field(min_length=1, max_length=40)]
    params: dict[str, float | str] = Field(default_factory=dict)


JobStatus = Literal["running", "finished", "failed", "stopped"]


class LabJob(BaseModel):
    job_id: str
    template: str
    # The Live Runs run this job creates, found by its name.
    run_name: str
    status: JobStatus
    exit_code: int | None
    started_at: float
    # The script's last output lines (errors, progress).
    output: list[str]


class LabJobList(BaseModel):
    jobs: list[LabJob]
