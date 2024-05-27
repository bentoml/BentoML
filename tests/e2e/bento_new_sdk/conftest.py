from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).parent.parent.parent.parent / "examples"
E2E_EXAMPLES = ["quickstart"]


@pytest.fixture(scope="package", autouse=True)
def prepare_models() -> None:
    pass


@pytest.fixture
def examples() -> Path:
    return EXAMPLE_DIR
