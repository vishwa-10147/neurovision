from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    service: str
    version: str
    environment: str
    # Largest model file the import endpoint accepts (NEUROVISION_MAX_UPLOAD_BYTES).
    max_upload_bytes: int
