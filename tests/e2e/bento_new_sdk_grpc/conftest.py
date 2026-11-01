from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def clear_import_cache() -> None:
    sys.modules.pop("service", None)
