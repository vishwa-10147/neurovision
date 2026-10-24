"""Neural Lab: list templates, launch bundled training scripts, track them."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, status

from app.core.errors import invalid, not_found
from app.schemas.lab import LabJob, LabJobList, LaunchRequest, TemplateList
from app.services.lab import TEMPLATES, LabError, LabManager

router = APIRouter(prefix="/lab", tags=["lab"])


def get_lab(request: Request) -> LabManager:
    lab: LabManager = request.app.state.lab
    return lab


Lab = Annotated[LabManager, Depends(get_lab)]


@router.get("/templates", response_model=TemplateList)
def list_templates(lab: Lab) -> TemplateList:
    return TemplateList(
        templates=[s.template for s in TEMPLATES.values()],
        enabled=lab.enabled,
        python=lab.python if lab.enabled else None,
    )


@router.post("/runs", status_code=status.HTTP_201_CREATED, response_model=LabJob)
def launch(body: LaunchRequest, request: Request, lab: Lab) -> LabJob:
    # The script reports back to this server; on the same machine, by IP.
    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    base_url = f"{request.url.scheme}://127.0.0.1:{port}"
    try:
        return lab.launch(body.template, body.params, base_url)
    except LabError as error:
        raise invalid("body", str(error)) from error


@router.get("/runs", response_model=LabJobList)
def list_jobs(lab: Lab) -> LabJobList:
    return LabJobList(jobs=lab.jobs())


@router.post("/runs/{job_id}/stop", response_model=LabJob)
def stop(job_id: Annotated[str, Path(max_length=40)], lab: Lab) -> LabJob:
    job = lab.stop(job_id)
    if job is None:
        raise not_found(f"Lab run '{job_id}' was not found.")
    return job
