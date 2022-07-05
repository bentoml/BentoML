from __future__ import annotations

from typing import TYPE_CHECKING

from ._internal.frameworks.onnxmlir import get
from ._internal.frameworks.onnxmlir import load_model
from ._internal.frameworks.onnxmlir import save_model
from ._internal.frameworks.onnxmlir import get_runnable
from ._internal.frameworks.onnxmlir import ONNXMLirOptions as ModelOptions  # type: ignore (unused imports)

if TYPE_CHECKING:
    from ._internal.frameworks.onnxmlir import ExecutionSession


__all__ = ["load_model", "save_model", "get", "get_runnable", "ExecutionSession"]
