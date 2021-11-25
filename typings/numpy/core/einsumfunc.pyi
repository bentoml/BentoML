import sys
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, overload

from numpy import _OrderKACF
from numpy.typing import (
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeUInt_co,
    _DTypeLikeBool,
    _DTypeLikeComplex,
    _DTypeLikeComplex_co,
    _DTypeLikeFloat,
    _DTypeLikeInt,
    _DTypeLikeUInt,
)

if sys.version_info >= (3, 8): ...
else: ...
_ArrayType = ...
_OptimizeKind = Union[None, bool, Literal["greedy", "optimal"], Sequence[Any]]
_CastingSafe = Literal["no", "equiv", "safe", "same_kind"]
_CastingUnsafe = Literal["unsafe"]
__all__: List[str]

@overload
def einsum(
    __subscripts: str,
    *operands: _ArrayLikeBool_co,
    out: None = ...,
    dtype: Optional[_DTypeLikeBool] = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...
) -> Any: ...
@overload
def einsum(
    __subscripts: str,
    *operands: _ArrayLikeUInt_co,
    out: None = ...,
    dtype: Optional[_DTypeLikeUInt] = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...
) -> Any: ...
@overload
def einsum(
    __subscripts: str,
    *operands: _ArrayLikeInt_co,
    out: None = ...,
    dtype: Optional[_DTypeLikeInt] = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...
) -> Any: ...
@overload
def einsum(
    __subscripts: str,
    *operands: _ArrayLikeFloat_co,
    out: None = ...,
    dtype: Optional[_DTypeLikeFloat] = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...
) -> Any: ...
@overload
def einsum(
    __subscripts: str,
    *operands: _ArrayLikeComplex_co,
    out: None = ...,
    dtype: Optional[_DTypeLikeComplex] = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...
) -> Any: ...
@overload
def einsum(
    __subscripts: str,
    *operands: Any,
    casting: _CastingUnsafe,
    dtype: Optional[_DTypeLikeComplex_co] = ...,
    out: None = ...,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = ...
) -> Any: ...
@overload
def einsum(
    __subscripts: str,
    *operands: _ArrayLikeComplex_co,
    out: _ArrayType,
    dtype: Optional[_DTypeLikeComplex_co] = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...
) -> _ArrayType: ...
@overload
def einsum(
    __subscripts: str,
    *operands: Any,
    out: _ArrayType,
    casting: _CastingUnsafe,
    dtype: Optional[_DTypeLikeComplex_co] = ...,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = ...
) -> _ArrayType: ...
def einsum_path(
    __subscripts: str, *operands: _ArrayLikeComplex_co, optimize: _OptimizeKind = ...
) -> Tuple[List[Any], str]: ...
