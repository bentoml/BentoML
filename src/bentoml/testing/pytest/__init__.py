from __future__ import annotations

from ..._internal.models import ModelContext

__all__ = ["TEST_MODEL_CONTEXT"]

TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing",
    framework_versions={"testing": "v1"},
)
