from __future__ import annotations

import importlib.util
from importlib.abc import MetaPathFinder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec
    from types import ModuleType
    from typing import Sequence


class FrameworkImporter(MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = ...
    ) -> ModuleSpec | None:
        if not fullname.startswith("bentoml."):
            return None
        framework = fullname.split(".")[1]
        if "." in framework:
            return None
        spec = importlib.util.find_spec(f"_bentoml_impl.frameworks.{framework}")
        if spec is None:
            spec = importlib.util.find_spec(f"bentoml._internal.frameworks.{framework}")
        return spec

    @classmethod
    def install(cls) -> None:
        import sys

        for finder in sys.meta_path:
            if isinstance(finder, cls):
                return

        sys.meta_path.append(cls())
