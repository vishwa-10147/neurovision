from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.core.errors import register_error_handlers
from app.main import app
from app.schemas.errors import ErrorResponse


def _error_app() -> FastAPI:
    """Small app with routes that exercise each error handler."""
    test_app = FastAPI()
    register_error_handlers(test_app)

    class Payload(BaseModel):
        width: int = Field(ge=1, le=64)

    @test_app.post("/validate")
    def validate(payload: Payload) -> dict[str, int]:
        return {"width": payload.width}

    @test_app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret path C:\\internal\\file.py")

    return test_app


def test_unknown_route_returns_error_envelope() -> None:
    response = TestClient(app).get("/api/v1/does-not-exist")
    assert response.status_code == 404
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code == "not_found"
    assert "/api/v1/does-not-exist" in body.error.message


def test_method_not_allowed_returns_error_envelope() -> None:
    response = TestClient(app).post("/api/v1/health")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_validation_error_lists_fields_without_echoing_input() -> None:
    response = TestClient(_error_app()).post("/validate", json={"width": 9999})
    assert response.status_code == 422
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code == "validation_error"
    assert [d.field for d in body.error.details] == ["body.width"]
    assert "9999" not in response.text


def test_unhandled_error_is_generic() -> None:
    client = TestClient(_error_app(), raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text
    assert "C:\\" not in response.text
