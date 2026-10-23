from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "neurovision-backend"
    assert set(body) == {
        "status",
        "service",
        "version",
        "environment",
        "max_upload_bytes",
    }
    assert body["max_upload_bytes"] == get_settings().max_upload_bytes


def test_health_allows_local_frontend_origin() -> None:
    response = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_health_rejects_unknown_origin() -> None:
    response = client.get("/api/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers
