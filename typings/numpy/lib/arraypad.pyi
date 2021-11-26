from typing import Any, Dict, List
from typing import Literal as L
from typing import Protocol, Tuple, TypeVar, overload
from numpy import dtype, generic, ndarray
from numpy.typing import (
    ArrayLike,
    NDArray,
    _ArrayLikeInt,
    _FiniteNestedSequence,
    _SupportsArray,
)

_SCT = TypeVar("_SCT", bound=generic)

class _ModeFunc(Protocol):
    def __call__(
        self,
        vector: NDArray[Any],
        iaxis_pad_width: Tuple[int, int],
        iaxis: int,
        kwargs: Dict[str, Any],
        /,
    ) -> None: ...

_ModeKind = L[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
    "empty",
]
_ArrayLike = _FiniteNestedSequence[_SupportsArray[dtype[_SCT]]]
__all__: List[str]

@overload
def pad(
    array: _ArrayLike[_SCT],
    pad_width: _ArrayLikeInt,
    mode: _ModeKind = ...,
    *,
    stat_length: None | _ArrayLikeInt = ...,
    constant_values: ArrayLike = ...,
    end_values: ArrayLike = ...,
    reflect_type: L["odd", "even"] = ...,
) -> NDArray[_SCT]: ...
@overload
def pad(
    array: ArrayLike,
    pad_width: _ArrayLikeInt,
    mode: _ModeKind = ...,
    *,
    stat_length: None | _ArrayLikeInt = ...,
    constant_values: ArrayLike = ...,
    end_values: ArrayLike = ...,
    reflect_type: L["odd", "even"] = ...,
) -> NDArray[Any]: ...
@overload
def pad(
    array: _ArrayLike[_SCT], pad_width: _ArrayLikeInt, mode: _ModeFunc, **kwargs: Any
) -> NDArray[_SCT]: ...
@overload
def pad(
    array: ArrayLike, pad_width: _ArrayLikeInt, mode: _ModeFunc, **kwargs: Any
) -> NDArray[Any]: ...
