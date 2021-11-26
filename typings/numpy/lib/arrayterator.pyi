from typing import Any, Generator, List, Tuple, TypeVar, Union, overload
from numpy import dtype, generic, ndarray
from numpy.typing import DTypeLike

_Shape = TypeVar("_Shape", bound=Any)
_DType = TypeVar("_DType", bound=dtype[Any])
_ScalarType = TypeVar("_ScalarType", bound=generic)
_Index = Union[Union[ellipsis, int, slice], Tuple[Union[ellipsis, int, slice], ...]]
__all__: List[str]

class Arrayterator(ndarray[_Shape, _DType]):
    var: ndarray[_Shape, _DType]
    buf_size: None | int
    start: List[int]
    stop: List[int]
    step: List[int]
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def flat(
        self: ndarray[Any, dtype[_ScalarType]]
    ) -> Generator[_ScalarType, None, None]: ...
    def __init__(
        self, var: ndarray[_Shape, _DType], buf_size: None | int = ...
    ) -> None: ...
    @overload
    def __array__(self, dtype: None = ...) -> ndarray[Any, _DType]: ...
    @overload
    def __array__(self, dtype: DTypeLike) -> ndarray[Any, dtype[Any]]: ...
    def __getitem__(self, index: _Index) -> Arrayterator[Any, _DType]: ...
    def __iter__(self) -> Generator[ndarray[Any, _DType], None, None]: ...
