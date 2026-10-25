from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def isolated_runs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Each test saves external runs to its own directory, never data/runs."""
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("NEUROVISION_RUNS_DIR", str(runs_dir))
    get_settings.cache_clear()
    yield runs_dir
    get_settings.cache_clear()
