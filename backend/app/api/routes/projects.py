from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, status

from app.schemas.projects import ProjectCreate, ProjectDocument, ProjectList
from app.services.project_store import ProjectStore

router = APIRouter(prefix="/projects", tags=["projects"])
ProjectId = Annotated[
    str, Path(min_length=1, max_length=64, pattern=r"^proj_[A-Za-z0-9]+$")
]


def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


Store = Annotated[ProjectStore, Depends(get_store)]


@router.get("", response_model=ProjectList)
def list_projects(store: Store) -> ProjectList:
    return ProjectList(projects=store.list())


@router.post("", response_model=ProjectDocument, status_code=status.HTTP_201_CREATED)
def save_project(body: ProjectCreate, store: Store) -> ProjectDocument:
    return store.save(body)


@router.get("/{project_id}", response_model=ProjectDocument)
def get_project(project_id: ProjectId, store: Store) -> ProjectDocument:
    return store.get(project_id)
