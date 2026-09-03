import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JULIA_TEST_DATA = REPO_ROOT / "test" / "data"
JULIA_BASELINE_OUTPUTS = REPO_ROOT / "test" / "baseline_outputs"


@pytest.fixture(scope="session")
def julia_test_data() -> Path:
    if not JULIA_TEST_DATA.exists():
        pytest.skip(f"Julia test data not found at {JULIA_TEST_DATA}")
    return JULIA_TEST_DATA


@pytest.fixture(scope="session")
def julia_available() -> bool:
    return shutil.which("julia") is not None
