from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from ._internal.models.model import ModelSignature
from ._internal.types import ModelSignatureDict

if TYPE_CHECKING:
    from _bentoml_sdk.types import Audio
    from _bentoml_sdk.types import File
    from _bentoml_sdk.types import Image
    from _bentoml_sdk.types import Tensor
    from _bentoml_sdk.types import Video
else:

    def __getattr__(name: str) -> None:
        if name not in ("File", "Image", "Audio", "Video", "Tensor"):
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

        import _bentoml_sdk.types

        return getattr(_bentoml_sdk.types, name)


__all__ = [
    "ModelSignature",
    "ModelSignatureDict",
    "File",
    "Image",
    "Audio",
    "Video",
    "Tensor",
    "Field",
]
