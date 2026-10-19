"""Generate shared/schemas/*.schema.json from the Pydantic models.

Run from backend/:  python -m app.schemas.export
A test fails when the committed files drift from the models.
"""

import json
from pathlib import Path
from typing import Any

from app.schemas.errors import ErrorResponse
from app.schemas.events import training_event_adapter
from app.schemas.graph import ModelGraph
from app.schemas.health import HealthResponse

SHARED_SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "shared" / "schemas"
_ID_BASE = "https://neurovision.local/schemas"


def _with_header(name: str, title: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{_ID_BASE}/{name}",
        **schema,
        "title": title,
    }


def build_schemas() -> dict[str, dict[str, Any]]:
    return {
        "model-graph.schema.json": _with_header(
            "model-graph.schema.json", "ModelGraph", ModelGraph.model_json_schema()
        ),
        "training-event.schema.json": _with_header(
            "training-event.schema.json",
            "TrainingEvent",
            training_event_adapter.json_schema(),
        ),
        "error.schema.json": _with_header(
            "error.schema.json", "ErrorResponse", ErrorResponse.model_json_schema()
        ),
        "health.schema.json": _with_header(
            "health.schema.json", "HealthResponse", HealthResponse.model_json_schema()
        ),
    }


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2) + "\n"


def main() -> None:
    for name, schema in build_schemas().items():
        path = SHARED_SCHEMAS_DIR / name
        path.write_text(render(schema), encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(SHARED_SCHEMAS_DIR.parents[1])}")


if __name__ == "__main__":
    main()
