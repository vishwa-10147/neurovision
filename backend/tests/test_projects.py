"""Versioned architecture project persistence."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.project_store import ProjectStore

CONFIG = {
    "name": "saved classifier",
    "input_units": 2,
    "hidden_layers": [{"units": 8, "activation": "relu"}],
    "output_units": 2,
    "output_activation": None,
    "seed": 7,
}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app()
    app.state.project_store = ProjectStore(tmp_path / "projects")
    with TestClient(app) as test_client:
        yield test_client


def test_save_list_and_reopen_project_after_store_recreation(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/api/v1/projects",
        json={
            "name": "saved classifier",
            "description": "first lesson",
            "config": CONFIG,
        },
    )
    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["project_id"].startswith("proj_")
    assert saved["schema_version"] == 1
    assert saved["config"] == CONFIG

    listed = client.get("/api/v1/projects")
    assert listed.status_code == 200
    assert listed.json()["projects"][0]["project_id"] == saved["project_id"]
    assert "config" not in listed.json()["projects"][0]

    project_dir = tmp_path / "projects"
    reopened = ProjectStore(project_dir).get(saved["project_id"])
    assert reopened.config.model_dump(mode="json") == CONFIG
    assert reopened.description == "first lesson"


def test_project_validation_rejects_unknown_fields(client: TestClient) -> None:
    response = client.post(
        "/api/v1/projects",
        json={"name": "bad", "config": CONFIG, "unexpected": True},
    )
    assert response.status_code == 422


def test_missing_project_is_a_safe_404(client: TestClient) -> None:
    response = client.get("/api/v1/projects/proj_missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
