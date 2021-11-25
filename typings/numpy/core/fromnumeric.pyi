import datetime as dt
import sys
from typing import Any, Literal, Optional, Sequence, Tuple, Union, overload

from numpy import (
    _ModeKind,
    _OrderACF,
    _OrderKACF,
    _PartitionKind,
    _SortKind,
    _SortSide,
    bool_,
    generic,
    intp,
    ndarray,
)
from numpy.typing import (
    ArrayLike,
    DTypeLike,
    _ArrayLikeBool_co,
    _ArrayLikeInt_co,
    _NumberLike_co,
    _Shape,
    _ShapeLike,
)

if sys.version_info >= (3, 8): ...
else: ...
_ScalarNumpy = Union[generic, dt.datetime, dt.timedelta]
_ScalarBuiltin = Union[str, bytes, dt.date, dt.timedelta, bool, int, float, complex]
_Scalar = Union[_ScalarBuiltin, _ScalarNumpy]
_ScalarGeneric = ...
_Number = ...

def take(
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
    mode: _ModeKind = ...,
) -> Any: ...
def reshape(a: ArrayLike, newshape: _ShapeLike, order: _OrderACF = ...) -> ndarray: ...
def choose(
    a: _ArrayLikeInt_co,
    choices: ArrayLike,
    out: Optional[ndarray] = ...,
    mode: _ModeKind = ...,
) -> Any: ...
def repeat(
    a: ArrayLike, repeats: _ArrayLikeInt_co, axis: Optional[int] = ...
) -> ndarray: ...
def put(
    a: ndarray, ind: _ArrayLikeInt_co, v: ArrayLike, mode: _ModeKind = ...
) -> None: ...
def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> ndarray: ...
def transpose(
    a: ArrayLike, axes: Union[None, Sequence[int], ndarray] = ...
) -> ndarray: ...
def partition(
    a: ArrayLike,
    kth: _ArrayLikeInt_co,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
def argpartition(
    a: ArrayLike,
    kth: _ArrayLikeInt_co,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> Any: ...
def sort(
    a: ArrayLike,
    axis: Optional[int] = ...,
    kind: Optional[_SortKind] = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
def argsort(
    a: ArrayLike,
    axis: Optional[int] = ...,
    kind: Optional[_SortKind] = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
@overload
def argmax(a: ArrayLike, axis: None = ..., out: Optional[ndarray] = ...) -> intp: ...
@overload
def argmax(
    a: ArrayLike, axis: Optional[int] = ..., out: Optional[ndarray] = ...
) -> Any: ...
@overload
def argmin(a: ArrayLike, axis: None = ..., out: Optional[ndarray] = ...) -> intp: ...
@overload
def argmin(
    a: ArrayLike, axis: Optional[int] = ..., out: Optional[ndarray] = ...
) -> Any: ...
@overload
def searchsorted(
    a: ArrayLike,
    v: _Scalar,
    side: _SortSide = ...,
    sorter: Optional[_ArrayLikeInt_co] = ...,
) -> intp: ...
@overload
def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    side: _SortSide = ...,
    sorter: Optional[_ArrayLikeInt_co] = ...,
) -> ndarray: ...
def resize(a: ArrayLike, new_shape: _ShapeLike) -> ndarray: ...
@overload
def squeeze(a: _ScalarGeneric, axis: Optional[_ShapeLike] = ...) -> _ScalarGeneric: ...
@overload
def squeeze(a: ArrayLike, axis: Optional[_ShapeLike] = ...) -> ndarray: ...
def diagonal(
    a: ArrayLike, offset: int = ..., axis1: int = ..., axis2: int = ...
) -> ndarray: ...
def trace(
    a: ArrayLike,
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
) -> Any: ...
def ravel(a: ArrayLike, order: _OrderKACF = ...) -> ndarray: ...
def nonzero(a: ArrayLike) -> Tuple[ndarray, ...]: ...
def shape(a: ArrayLike) -> _Shape: ...
def compress(
    condition: ArrayLike,
    a: ArrayLike,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
) -> ndarray: ...
@overload
def clip(
    a: ArrayLike,
    a_min: ArrayLike,
    a_max: Optional[ArrayLike],
    out: Optional[ndarray] = ...,
    **kwargs: Any
) -> Any: ...
@overload
def clip(
    a: ArrayLike,
    a_min: None,
    a_max: ArrayLike,
    out: Optional[ndarray] = ...,
    **kwargs: Any
) -> Any: ...
def sum(
    a: ArrayLike,
    axis: _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...
@overload
def all(
    a: ArrayLike, axis: None = ..., out: None = ..., keepdims: Literal[False] = ...
) -> bool_: ...
@overload
def all(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Any: ...
@overload
def any(
    a: ArrayLike, axis: None = ..., out: None = ..., keepdims: Literal[False] = ...
) -> bool_: ...
@overload
def any(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Any: ...
def cumsum(
    a: ArrayLike,
    axis: Optional[int] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
) -> ndarray: ...
def ptp(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Any: ...
def amax(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...
def amin(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...
def prod(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...
def cumprod(
    a: ArrayLike,
    axis: Optional[int] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
) -> ndarray: ...
def ndim(a: ArrayLike) -> int: ...
def size(a: ArrayLike, axis: Optional[int] = ...) -> int: ...
def around(a: ArrayLike, decimals: int = ..., out: Optional[ndarray] = ...) -> Any: ...
def mean(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Any: ...
def std(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    ddof: int = ...,
    keepdims: bool = ...,
) -> Any: ...
def var(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    ddof: int = ...,
    keepdims: bool = ...,
) -> Any: ...
