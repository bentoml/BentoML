from typing import Any, Callable, List, Sequence, Tuple, Type, TypeVar, Union, overload
from numpy import (
    _OrderCF,
    bool_,
    complexfloating,
    datetime64,
    dtype,
    float64,
    floating,
    generic,
    int_,
    intp,
    ndarray,
    number,
    object_,
    signedinteger,
    timedelta64,
)
from numpy.typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _FiniteNestedSequence,
    _SupportsArray,
    _SupportsDType,
)

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=generic)
_MaskFunc = Callable[
    [NDArray[int_], _T],
    NDArray[Union[number[Any], bool_, timedelta64, datetime64, object_]],
]
_DTypeLike = Union[Type[_SCT], dtype[_SCT], _SupportsDType[dtype[_SCT]]]
_ArrayLike = _FiniteNestedSequence[_SupportsArray[dtype[_SCT]]]
__all__: List[str]

@overload
def fliplr(m: _ArrayLike[_SCT]) -> NDArray[_SCT]: ...
@overload
def fliplr(m: ArrayLike) -> NDArray[Any]: ...
@overload
def flipud(m: _ArrayLike[_SCT]) -> NDArray[_SCT]: ...
@overload
def flipud(m: ArrayLike) -> NDArray[Any]: ...
@overload
def eye(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    like: None | ArrayLike = ...
) -> NDArray[float64]: ...
@overload
def eye(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: _DTypeLike[_SCT] = ...,
    order: _OrderCF = ...,
    *,
    like: None | ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def eye(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: None | ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def diag(v: _ArrayLike[_SCT], k: int = ...) -> NDArray[_SCT]: ...
@overload
def diag(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...
@overload
def diagflat(v: _ArrayLike[_SCT], k: int = ...) -> NDArray[_SCT]: ...
@overload
def diagflat(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...
@overload
def tri(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: None = ...,
    *,
    like: None | ArrayLike = ...
) -> NDArray[float64]: ...
@overload
def tri(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: _DTypeLike[_SCT] = ...,
    *,
    like: None | ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def tri(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: DTypeLike = ...,
    *,
    like: None | ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def tril(v: _ArrayLike[_SCT], k: int = ...) -> NDArray[_SCT]: ...
@overload
def tril(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...
@overload
def triu(v: _ArrayLike[_SCT], k: int = ...) -> NDArray[_SCT]: ...
@overload
def triu(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...
@overload
def vander(
    x: _ArrayLikeInt_co, N: None | int = ..., increasing: bool = ...
) -> NDArray[signedinteger[Any]]: ...
@overload
def vander(
    x: _ArrayLikeFloat_co, N: None | int = ..., increasing: bool = ...
) -> NDArray[floating[Any]]: ...
@overload
def vander(
    x: _ArrayLikeComplex_co, N: None | int = ..., increasing: bool = ...
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def vander(
    x: _ArrayLikeObject_co, N: None | int = ..., increasing: bool = ...
) -> NDArray[object_]: ...
@overload
def histogram2d(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    bins: int | Sequence[int] = ...,
    range: None | _ArrayLikeFloat_co = ...,
    normed: None | bool = ...,
    weights: None | _ArrayLikeFloat_co = ...,
    density: None | bool = ...,
) -> Tuple[NDArray[float64], NDArray[floating[Any]], NDArray[floating[Any]]]: ...
@overload
def histogram2d(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    bins: int | Sequence[int] = ...,
    range: None | _ArrayLikeFloat_co = ...,
    normed: None | bool = ...,
    weights: None | _ArrayLikeFloat_co = ...,
    density: None | bool = ...,
) -> Tuple[
    NDArray[float64],
    NDArray[complexfloating[Any, Any]],
    NDArray[complexfloating[Any, Any]],
]: ...
@overload
def histogram2d(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    bins: Sequence[_ArrayLikeInt_co],
    range: None | _ArrayLikeFloat_co = ...,
    normed: None | bool = ...,
    weights: None | _ArrayLikeFloat_co = ...,
    density: None | bool = ...,
) -> Tuple[NDArray[float64], NDArray[Any], NDArray[Any]]: ...
@overload
def mask_indices(
    n: int, mask_func: _MaskFunc[int], k: int = ...
) -> Tuple[NDArray[intp], NDArray[intp]]: ...
@overload
def mask_indices(
    n: int, mask_func: _MaskFunc[_T], k: _T
) -> Tuple[NDArray[intp], NDArray[intp]]: ...
def tril_indices(
    n: int, k: int = ..., m: None | int = ...
) -> Tuple[NDArray[int_], NDArray[int_]]: ...
def tril_indices_from(
    arr: NDArray[Any], k: int = ...
) -> Tuple[NDArray[int_], NDArray[int_]]: ...
def triu_indices(
    n: int, k: int = ..., m: None | int = ...
) -> Tuple[NDArray[int_], NDArray[int_]]: ...
def triu_indices_from(
    arr: NDArray[Any], k: int = ...
) -> Tuple[NDArray[int_], NDArray[int_]]: ...
