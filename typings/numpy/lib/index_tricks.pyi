from typing import (
    Any,
    Generic,
    List,
    Literal,
    Sequence,
    SupportsIndex,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from numpy import (
    _ModeKind,
    _OrderCF,
    bool_,
    bytes_,
    complex_,
    dtype,
    float_,
    int_,
    integer,
    intp,
)
from numpy import matrix as _Matrix
from numpy import ndarray
from numpy import ndenumerate as ndenumerate
from numpy import ndindex as ndindex
from numpy import str_
from numpy.core.multiarray import ravel_multi_index as ravel_multi_index
from numpy.core.multiarray import unravel_index as unravel_index
from numpy.typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLikeInt,
    _FiniteNestedSequence,
    _NestedSequence,
    _ShapeLike,
    _SupportsDType,
)

_T = TypeVar("_T")
_DType = TypeVar("_DType", bound=dtype[Any])
_BoolType = TypeVar("_BoolType", Literal[True], Literal[False])
_TupType = TypeVar("_TupType", bound=Tuple[Any, ...])
_ArrayType = TypeVar("_ArrayType", bound=ndarray[Any, Any])
__all__: List[str]

@overload
def ix_(
    *args: _FiniteNestedSequence[_SupportsDType[_DType]],
) -> Tuple[ndarray[Any, _DType], ...]: ...
@overload
def ix_(*args: str | _NestedSequence[str]) -> Tuple[NDArray[str_], ...]: ...
@overload
def ix_(*args: bytes | _NestedSequence[bytes]) -> Tuple[NDArray[bytes_], ...]: ...
@overload
def ix_(*args: bool | _NestedSequence[bool]) -> Tuple[NDArray[bool_], ...]: ...
@overload
def ix_(*args: int | _NestedSequence[int]) -> Tuple[NDArray[int_], ...]: ...
@overload
def ix_(*args: float | _NestedSequence[float]) -> Tuple[NDArray[float_], ...]: ...
@overload
def ix_(*args: complex | _NestedSequence[complex]) -> Tuple[NDArray[complex_], ...]: ...

class nd_grid(Generic[_BoolType]):
    sparse: _BoolType
    def __init__(self, sparse: _BoolType = ...) -> None: ...
    @overload
    def __getitem__(
        self: nd_grid[Literal[False]], key: Union[slice, Sequence[slice]]
    ) -> NDArray[Any]: ...
    @overload
    def __getitem__(
        self: nd_grid[Literal[True]], key: Union[slice, Sequence[slice]]
    ) -> List[NDArray[Any]]: ...

class MGridClass(nd_grid[Literal[False]]):
    def __init__(self) -> None: ...

mgrid: MGridClass

class OGridClass(nd_grid[Literal[True]]):
    def __init__(self) -> None: ...

ogrid: OGridClass

class AxisConcatenator:
    axis: int
    matrix: bool
    ndmin: int
    trans1d: int
    def __init__(
        self, axis: int = ..., matrix: bool = ..., ndmin: int = ..., trans1d: int = ...
    ) -> None: ...
    @staticmethod
    @overload
    def concatenate(
        *a: ArrayLike, axis: SupportsIndex = ..., out: None = ...
    ) -> NDArray[Any]: ...
    @staticmethod
    @overload
    def concatenate(
        *a: ArrayLike, axis: SupportsIndex = ..., out: _ArrayType = ...
    ) -> _ArrayType: ...
    @staticmethod
    def makemat(
        data: ArrayLike, dtype: DTypeLike = ..., copy: bool = ...
    ) -> _Matrix: ...
    def __getitem__(self, key: Any) -> Any: ...

class RClass(AxisConcatenator):
    axis: Literal[0]
    matrix: Literal[False]
    ndmin: Literal[1]
    trans1d: Literal[-1]
    def __init__(self) -> None: ...

r_: RClass

class CClass(AxisConcatenator):
    axis: Literal[-1]
    matrix: Literal[False]
    ndmin: Literal[2]
    trans1d: Literal[0]
    def __init__(self) -> None: ...

c_: CClass

class IndexExpression(Generic[_BoolType]):
    maketuple: _BoolType
    def __init__(self, maketuple: _BoolType) -> None: ...
    @overload
    def __getitem__(self, item: _TupType) -> _TupType: ...
    @overload
    def __getitem__(self: IndexExpression[Literal[True]], item: _T) -> Tuple[_T]: ...
    @overload
    def __getitem__(self: IndexExpression[Literal[False]], item: _T) -> _T: ...

index_exp: IndexExpression[Literal[True]]
s_: IndexExpression[Literal[False]]

def fill_diagonal(a: ndarray[Any, Any], val: Any, wrap: bool = ...) -> None: ...
def diag_indices(n: int, ndim: int = ...) -> Tuple[NDArray[int_], ...]: ...
def diag_indices_from(arr: ArrayLike) -> Tuple[NDArray[int_], ...]: ...
