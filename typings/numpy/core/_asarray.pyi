
import sys
from typing import Iterable, Literal, TypeVar, Union, overload

from numpy import _OrderKACF, ndarray
from numpy.typing import ArrayLike, DTypeLike

if sys.version_info >= (3, 8):
    ...
else:
    ...
_ArrayType = TypeVar("_ArrayType", bound=ndarray)
def asarray(a: object, dtype: DTypeLike = ..., order: _OrderKACF = ..., *, like: ArrayLike = ...) -> ndarray:
    ...

@overload
def asanyarray(a: _ArrayType, dtype: None = ..., order: _OrderKACF = ..., *, like: ArrayLike = ...) -> _ArrayType:
    ...

@overload
def asanyarray(a: object, dtype: DTypeLike = ..., order: _OrderKACF = ..., *, like: ArrayLike = ...) -> ndarray:
    ...

def ascontiguousarray(a: object, dtype: DTypeLike = ..., *, like: ArrayLike = ...) -> ndarray:
    ...

def asfortranarray(a: object, dtype: DTypeLike = ..., *, like: ArrayLike = ...) -> ndarray:
    ...

_Requirements = Literal["C", "C_CONTIGUOUS", "CONTIGUOUS", "F", "F_CONTIGUOUS", "FORTRAN", "A", "ALIGNED", "W", "WRITEABLE", "O", "OWNDATA"]
_E = Literal["E", "ENSUREARRAY"]
_RequirementsWithE = Union[_Requirements, _E]
@overload
def require(a: _ArrayType, dtype: None = ..., requirements: Union[None, _Requirements, Iterable[_Requirements]] = ..., *, like: ArrayLike = ...) -> _ArrayType:
    ...

@overload
def require(a: object, dtype: DTypeLike = ..., requirements: Union[_E, Iterable[_RequirementsWithE]] = ..., *, like: ArrayLike = ...) -> ndarray:
    ...

@overload
def require(a: object, dtype: DTypeLike = ..., requirements: Union[None, _Requirements, Iterable[_Requirements]] = ..., *, like: ArrayLike = ...) -> ndarray:
    ...

