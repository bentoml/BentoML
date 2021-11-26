from typing import Any, Dict, Iterable, List, SupportsIndex, TypeVar, overload
from numpy import dtype, generic
from numpy.typing import (
    ArrayLike,
    NDArray,
    _FiniteNestedSequence,
    _Shape,
    _ShapeLike,
    _SupportsArray,
)

_SCT = TypeVar("_SCT", bound=generic)
_ArrayLike = _FiniteNestedSequence[_SupportsArray[dtype[_SCT]]]
__all__: List[str]

class DummyArray:
    __array_interface__: Dict[str, Any]
    base: None | NDArray[Any]
    def __init__(
        self, interface: Dict[str, Any], base: None | NDArray[Any] = ...
    ) -> None: ...

@overload
def as_strided(
    x: _ArrayLike[_SCT],
    shape: None | Iterable[int] = ...,
    strides: None | Iterable[int] = ...,
    subok: bool = ...,
    writeable: bool = ...,
) -> NDArray[_SCT]: ...
@overload
def as_strided(
    x: ArrayLike,
    shape: None | Iterable[int] = ...,
    strides: None | Iterable[int] = ...,
    subok: bool = ...,
    writeable: bool = ...,
) -> NDArray[Any]: ...
@overload
def sliding_window_view(
    x: _ArrayLike[_SCT],
    window_shape: int | Iterable[int],
    axis: None | SupportsIndex = ...,
    *,
    subok: bool = ...,
    writeable: bool = ...
) -> NDArray[_SCT]: ...
@overload
def sliding_window_view(
    x: ArrayLike,
    window_shape: int | Iterable[int],
    axis: None | SupportsIndex = ...,
    *,
    subok: bool = ...,
    writeable: bool = ...
) -> NDArray[Any]: ...
@overload
def broadcast_to(
    array: _ArrayLike[_SCT], shape: int | Iterable[int], subok: bool = ...
) -> NDArray[_SCT]: ...
@overload
def broadcast_to(
    array: ArrayLike, shape: int | Iterable[int], subok: bool = ...
) -> NDArray[Any]: ...
def broadcast_shapes(*args: _ShapeLike) -> _Shape: ...
def broadcast_arrays(*args: ArrayLike, subok: bool = ...) -> List[NDArray[Any]]: ...
