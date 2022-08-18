from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._internal.types import AnyType
    from ._internal.models.model import ModelSignature

    class ModelSignatureDict(t.TypedDict, total=False):
        batchable: bool
        batch_dim: tuple[int, int] | int | None
        input_spec: tuple[AnyType] | AnyType | None
        output_spec: AnyType | None

    __all__ = ["ModelSignature", "ModelSignatureDict"]
