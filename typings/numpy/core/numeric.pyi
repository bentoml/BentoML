from typing import (
    Any,
    Callable,
    List,
    Literal,
    NoReturn,
    Sequence,
    SupportsAbs,
    SupportsIndex,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from numpy import ComplexWarning as ComplexWarning
from numpy import (
    _OrderCF,
    _OrderKACF,
    bool_,
    complexfloating,
    dtype,
    float64,
    floating,
    generic,
    int_,
    intp,
    object_,
    signedinteger,
    timedelta64,
    unsignedinteger,
)
from numpy.typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
    _FiniteNestedSequence,
    _ScalarLike_co,
    _ShapeLike,
    _SupportsArray,
    _SupportsDType,
)
from typing_extensions import TypeGuard

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=generic)
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])
_DTypeLike = Union[dtype[_SCT], Type[_SCT], _SupportsDType[dtype[_SCT]]]
_ArrayLike = _FiniteNestedSequence[_SupportsArray[dtype[_SCT]]]
_CorrelateMode = Literal["valid", "same", "full"]
__all__: List[str]

@overload
def zeros_like(
    a: _ArrayType,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: Literal[True] = ...,
    shape: None = ...,
) -> _ArrayType: ...
@overload
def zeros_like(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[_SCT]: ...
@overload
def zeros_like(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[Any]: ...
@overload
def zeros_like(
    a: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[_SCT]: ...
@overload
def zeros_like(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[Any]: ...
@overload
def ones(
    shape: _ShapeLike,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...
) -> NDArray[float64]: ...
@overload
def ones(
    shape: _ShapeLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def ones(
    shape: _ShapeLike, dtype: DTypeLike, order: _OrderCF = ..., *, like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def ones_like(
    a: _ArrayType,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: Literal[True] = ...,
    shape: None = ...,
) -> _ArrayType: ...
@overload
def ones_like(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[_SCT]: ...
@overload
def ones_like(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[Any]: ...
@overload
def ones_like(
    a: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[_SCT]: ...
@overload
def ones_like(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[Any]: ...
@overload
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: DTypeLike,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...
) -> NDArray[Any]: ...
@overload
def full_like(
    a: _ArrayType,
    fill_value: Any,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: Literal[True] = ...,
    shape: None = ...,
) -> _ArrayType: ...
@overload
def full_like(
    a: _ArrayLike[_SCT],
    fill_value: Any,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[_SCT]: ...
@overload
def full_like(
    a: object,
    fill_value: Any,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[Any]: ...
@overload
def full_like(
    a: Any,
    fill_value: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[_SCT]: ...
@overload
def full_like(
    a: Any,
    fill_value: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
) -> NDArray[Any]: ...
@overload
def count_nonzero(
    a: ArrayLike, axis: None = ..., *, keepdims: Literal[False] = ...
) -> int: ...
@overload
def count_nonzero(
    a: ArrayLike, axis: _ShapeLike = ..., *, keepdims: bool = ...
) -> Any: ...
def isfortran(a: NDArray[Any] | generic) -> bool: ...
def argwhere(a: ArrayLike) -> NDArray[intp]: ...
def flatnonzero(a: ArrayLike) -> NDArray[intp]: ...
@overload
def correlate(
    a: _ArrayLikeBool_co, v: _ArrayLikeBool_co, mode: _CorrelateMode = ...
) -> NDArray[bool_]: ...
@overload
def correlate(
    a: _ArrayLikeUInt_co, v: _ArrayLikeUInt_co, mode: _CorrelateMode = ...
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def correlate(
    a: _ArrayLikeInt_co, v: _ArrayLikeInt_co, mode: _CorrelateMode = ...
) -> NDArray[signedinteger[Any]]: ...
@overload
def correlate(
    a: _ArrayLikeFloat_co, v: _ArrayLikeFloat_co, mode: _CorrelateMode = ...
) -> NDArray[floating[Any]]: ...
@overload
def correlate(
    a: _ArrayLikeComplex_co, v: _ArrayLikeComplex_co, mode: _CorrelateMode = ...
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def correlate(
    a: _ArrayLikeTD64_co, v: _ArrayLikeTD64_co, mode: _CorrelateMode = ...
) -> NDArray[timedelta64]: ...
@overload
def correlate(
    a: _ArrayLikeObject_co, v: _ArrayLikeObject_co, mode: _CorrelateMode = ...
) -> NDArray[object_]: ...
@overload
def convolve(
    a: _ArrayLikeBool_co, v: _ArrayLikeBool_co, mode: _CorrelateMode = ...
) -> NDArray[bool_]: ...
@overload
def convolve(
    a: _ArrayLikeUInt_co, v: _ArrayLikeUInt_co, mode: _CorrelateMode = ...
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def convolve(
    a: _ArrayLikeInt_co, v: _ArrayLikeInt_co, mode: _CorrelateMode = ...
) -> NDArray[signedinteger[Any]]: ...
@overload
def convolve(
    a: _ArrayLikeFloat_co, v: _ArrayLikeFloat_co, mode: _CorrelateMode = ...
) -> NDArray[floating[Any]]: ...
@overload
def convolve(
    a: _ArrayLikeComplex_co, v: _ArrayLikeComplex_co, mode: _CorrelateMode = ...
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def convolve(
    a: _ArrayLikeTD64_co, v: _ArrayLikeTD64_co, mode: _CorrelateMode = ...
) -> NDArray[timedelta64]: ...
@overload
def convolve(
    a: _ArrayLikeObject_co, v: _ArrayLikeObject_co, mode: _CorrelateMode = ...
) -> NDArray[object_]: ...
@overload
def outer(
    a: _ArrayLikeBool_co, b: _ArrayLikeBool_co, out: None = ...
) -> NDArray[bool_]: ...
@overload
def outer(
    a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co, out: None = ...
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def outer(
    a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, out: None = ...
) -> NDArray[signedinteger[Any]]: ...
@overload
def outer(
    a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, out: None = ...
) -> NDArray[floating[Any]]: ...
@overload
def outer(
    a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, out: None = ...
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def outer(
    a: _ArrayLikeTD64_co, b: _ArrayLikeTD64_co, out: None = ...
) -> NDArray[timedelta64]: ...
@overload
def outer(
    a: _ArrayLikeObject_co, b: _ArrayLikeObject_co, out: None = ...
) -> NDArray[object_]: ...
@overload
def outer(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    b: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    out: _ArrayType,
) -> _ArrayType: ...
@overload
def tensordot(
    a: _ArrayLikeBool_co,
    b: _ArrayLikeBool_co,
    axes: int | Tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[bool_]: ...
@overload
def tensordot(
    a: _ArrayLikeUInt_co,
    b: _ArrayLikeUInt_co,
    axes: int | Tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def tensordot(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axes: int | Tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def tensordot(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axes: int | Tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[floating[Any]]: ...
@overload
def tensordot(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axes: int | Tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def tensordot(
    a: _ArrayLikeTD64_co,
    b: _ArrayLikeTD64_co,
    axes: int | Tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[timedelta64]: ...
@overload
def tensordot(
    a: _ArrayLikeObject_co,
    b: _ArrayLikeObject_co,
    axes: int | Tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[object_]: ...
@overload
def roll(
    a: _ArrayLike[_SCT], shift: _ShapeLike, axis: None | _ShapeLike = ...
) -> NDArray[_SCT]: ...
@overload
def roll(
    a: ArrayLike, shift: _ShapeLike, axis: None | _ShapeLike = ...
) -> NDArray[Any]: ...
def rollaxis(a: NDArray[_SCT], axis: int, start: int = ...) -> NDArray[_SCT]: ...
def moveaxis(
    a: NDArray[_SCT], source: _ShapeLike, destination: _ShapeLike
) -> NDArray[_SCT]: ...
@overload
def cross(
    a: _ArrayLikeBool_co,
    b: _ArrayLikeBool_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NoReturn: ...
@overload
def cross(
    a: _ArrayLikeUInt_co,
    b: _ArrayLikeUInt_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def cross(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def cross(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NDArray[floating[Any]]: ...
@overload
def cross(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def cross(
    a: _ArrayLikeObject_co,
    b: _ArrayLikeObject_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NDArray[object_]: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: Type[int] = ..., sparse: Literal[False] = ...
) -> NDArray[int_]: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: Type[int] = ..., sparse: Literal[True] = ...
) -> Tuple[NDArray[int_], ...]: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: _DTypeLike[_SCT], sparse: Literal[False] = ...
) -> NDArray[_SCT]: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: _DTypeLike[_SCT], sparse: Literal[True]
) -> Tuple[NDArray[_SCT], ...]: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: DTypeLike, sparse: Literal[False] = ...
) -> NDArray[Any]: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: DTypeLike, sparse: Literal[True]
) -> Tuple[NDArray[Any], ...]: ...
def fromfunction(
    function: Callable[..., _T],
    shape: Sequence[int],
    *,
    dtype: DTypeLike = ...,
    like: ArrayLike = ...,
    **kwargs: Any
) -> _T: ...
def isscalar(
    element: object,
) -> TypeGuard[generic | bool | int | float | complex | str | bytes | memoryview]: ...
def binary_repr(num: int, width: None | int = ...) -> str: ...
def base_repr(
    number: SupportsAbs[float], base: float = ..., padding: SupportsIndex = ...
) -> str: ...
@overload
def identity(
    n: int, dtype: None = ..., *, like: ArrayLike = ...
) -> NDArray[float64]: ...
@overload
def identity(
    n: int, dtype: _DTypeLike[_SCT], *, like: ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def identity(n: int, dtype: DTypeLike, *, like: ArrayLike = ...) -> NDArray[Any]: ...
def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> bool: ...
@overload
def isclose(
    a: _ScalarLike_co,
    b: _ScalarLike_co,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> bool_: ...
@overload
def isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> NDArray[bool_]: ...
def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: bool = ...) -> bool: ...
def array_equiv(a1: ArrayLike, a2: ArrayLike) -> bool: ...
