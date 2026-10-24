"""POST /classical/fit: fit one classical algorithm for the Classical ML Lab."""

from fastapi import APIRouter

from app.core.errors import invalid
from app.schemas.classical import ClassicRequest, ClassicResult
from app.services.classical import ClassicError, fit

router = APIRouter(prefix="/classical", tags=["classical"])


# Plain `def`: fitting runs in FastAPI's thread pool, off the event loop.
@router.post("/fit", response_model=ClassicResult)
def fit_classical(body: ClassicRequest) -> ClassicResult:
    try:
        return fit(body)
    except ClassicError as error:
        raise invalid("body.algorithm", str(error)) from error
