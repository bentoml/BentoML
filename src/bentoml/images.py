from typing import TYPE_CHECKING
from typing import Any

from _bentoml_sdk.images import Image
from bentoml._internal.utils import warn_deprecated

if TYPE_CHECKING:
    PythonImage = Image


def __getattr__(name: str) -> Any:
    if name == "PythonImage":
        warn_deprecated("PythonImage is an alias of Image, use Image instead")
        return Image
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["Image"]
