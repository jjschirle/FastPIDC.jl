import shutil
import subprocess
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


@pytest.fixture(scope="session")
def run_julia(julia_available):
    """Run a Julia script against the repository's FastPIDC.jl project.

    Skips the calling test if no `julia` executable is on PATH, and fails it
    (rather than returning partial output) if the script errors.
    """

    def _run(script: str, timeout: float = 600) -> str:
        if not julia_available:
            pytest.skip("julia executable not found")
        # Julia formats warnings with box-drawing characters, so decode
        # explicitly rather than relying on the (possibly ASCII) locale.
        proc = subprocess.run(
            ["julia", f"--project={REPO_ROOT}", "-e", script],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if proc.returncode != 0:
            pytest.fail(f"julia run failed:\n{proc.stdout}\n{proc.stderr}")
        return proc.stdout

    return _run
