from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._internal.models.model import ModelSignature
    from ._internal.models.model import ModelSignatureDict

    __all__ = ["ModelSignature", "ModelSignatureDict"]
