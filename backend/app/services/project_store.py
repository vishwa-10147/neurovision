"""Atomic, versioned local storage for architecture projects.

This is intentionally a filesystem-backed development store. It keeps the
project format explicit while avoiding a database before multi-user
ownership/authentication exists.
"""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.core.errors import not_found
from app.schemas.projects import (
    PROJECT_SCHEMA_VERSION,
    ProjectCreate,
    ProjectDocument,
    ProjectSummary,
)


class ProjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def list(self) -> list[ProjectSummary]:
        documents: list[ProjectSummary] = []
        for path in self.root.glob("proj_*.json"):
            try:
                documents.append(ProjectSummary.model_validate(self._read(path)))
            except (OSError, ValueError):
                # A malformed file should not make every valid project disappear.
                continue
        return sorted(documents, key=lambda project: project.updated_at, reverse=True)

    def get(self, project_id: str) -> ProjectDocument:
        path = self._path(project_id)
        if not path.exists():
            raise not_found(f"Project '{project_id}' was not found.")
        try:
            return ProjectDocument.model_validate(self._read(path))
        except (OSError, ValueError) as error:
            raise not_found(f"Project '{project_id}' could not be read.") from error

    def save(self, request: ProjectCreate) -> ProjectDocument:
        now = datetime.now(UTC).isoformat()
        project_id = f"proj_{uuid4().hex[:12]}"
        document = ProjectDocument(
            project_id=project_id,
            name=request.name,
            description=request.description,
            schema_version=PROJECT_SCHEMA_VERSION,
            created_at=now,
            updated_at=now,
            config=request.config,
        )
        self._atomic_write(self._path(project_id), document.model_dump(mode="json"))
        return document

    def _path(self, project_id: str) -> Path:
        # IDs are validated by the API model; this extra check protects direct
        # service callers from escaping the configured project directory.
        if not project_id.startswith("proj_") or Path(project_id).name != project_id:
            raise not_found(f"Project '{project_id}' was not found.")
        return self.root / f"{project_id}.json"

    @staticmethod
    def _read(path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))

    def _atomic_write(self, path: Path, payload: object) -> None:
        temporary = path.with_suffix(".tmp")
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with self._lock:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
