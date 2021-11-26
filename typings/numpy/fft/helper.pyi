from typing import Any, List, TypeVar, overload
from numpy import complexfloating, dtype, floating, generic, integer
from numpy.typing import (
    ArrayLike,
    NDArray,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _FiniteNestedSequence,
    _ShapeLike,
    _SupportsArray,
)

_SCT = TypeVar("_SCT", bound=generic)
_ArrayLike = _FiniteNestedSequence[_SupportsArray[dtype[_SCT]]]
__all__: List[str]

@overload
def fftshift(x: _ArrayLike[_SCT], axes: None | _ShapeLike = ...) -> NDArray[_SCT]: ...
@overload
def fftshift(x: ArrayLike, axes: None | _ShapeLike = ...) -> NDArray[Any]: ...
@overload
def ifftshift(x: _ArrayLike[_SCT], axes: None | _ShapeLike = ...) -> NDArray[_SCT]: ...
@overload
def ifftshift(x: ArrayLike, axes: None | _ShapeLike = ...) -> NDArray[Any]: ...
@overload
def fftfreq(n: int | integer[Any], d: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
@overload
def fftfreq(
    n: int | integer[Any], d: _ArrayLikeComplex_co
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def rfftfreq(
    n: int | integer[Any], d: _ArrayLikeFloat_co
) -> NDArray[floating[Any]]: ...
@overload
def rfftfreq(
    n: int | integer[Any], d: _ArrayLikeComplex_co
) -> NDArray[complexfloating[Any, Any]]: ...
