import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).parent.parent.parent.parent / "examples"
E2E_EXAMPLES = ["quickstart"]


@pytest.fixture(scope="package", autouse=True)
def prepare_models() -> None:
    for example in E2E_EXAMPLES:
        subprocess.run(
            [sys.executable, str(EXAMPLE_DIR / example / "prepare_model.py")],
            check=True,
            cwd=str(EXAMPLE_DIR / example),
        )


@pytest.fixture
def examples() -> Path:
    return EXAMPLE_DIR
