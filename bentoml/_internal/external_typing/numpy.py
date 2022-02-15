from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typing as t

    from numpy import generic as NpGeneric
    from numpy.typing import NDArray
    from numpy.typing import DTypeLike as NpDTypeLike

    NpNDArray = NDArray[t.Any]

    __all__ = ["NpNDArray", "NpGeneric", "NpDTypeLike"]
