import ctypes
import os
import sys
from ctypes import c_int64 as _c_intp
from typing import Any, ClassVar, Dict, Generic, Iterable, List
from typing import Literal as L
from typing import Optional, Sequence, Tuple, Type, TypeVar, Union, overload
from numpy import (
    bool_,
    byte,
    double,
    dtype,
    float_,
    generic,
    int_,
    intc,
    longdouble,
    longlong,
    ndarray,
    short,
    single,
    ubyte,
    uint,
    uintc,
    ulonglong,
    ushort,
    void,
)
from numpy.core._internal import _ctypes
from numpy.core.multiarray import flagsobj
from numpy.typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _BoolCodes,
    _ByteCodes,
    _DoubleCodes,
    _FiniteNestedSequence,
    _IntCCodes,
    _IntCodes,
    _LongDoubleCodes,
    _LongLongCodes,
    _ShapeLike,
    _ShortCodes,
    _SingleCodes,
    _SupportsArray,
    _SupportsDType,
    _UByteCodes,
    _UIntCCodes,
    _UIntCodes,
    _ULongLongCodes,
    _UShortCodes,
    _VoidDTypeLike,
)

_DType = TypeVar("_DType", bound=dtype[Any])
_DTypeOptional = TypeVar("_DTypeOptional", bound=Optional[dtype[Any]])
_SCT = TypeVar("_SCT", bound=generic)
_DTypeLike = Union[dtype[_SCT], Type[_SCT], _SupportsDType[dtype[_SCT]]]
_ArrayLike = _FiniteNestedSequence[_SupportsArray[dtype[_SCT]]]
_FlagsKind = L[
    "C_CONTIGUOUS",
    "CONTIGUOUS",
    "C",
    "F_CONTIGUOUS",
    "FORTRAN",
    "F",
    "ALIGNED",
    "A",
    "WRITEABLE",
    "W",
    "OWNDATA",
    "O",
    "UPDATEIFCOPY",
    "U",
    "WRITEBACKIFCOPY",
    "X",
]

class _ndptr(ctypes.c_void_p, Generic[_DTypeOptional]):
    _dtype_: ClassVar[_DTypeOptional]
    _shape_: ClassVar[None]
    _ndim_: ClassVar[None | int]
    _flags_: ClassVar[None | List[_FlagsKind]]
    @overload
    @classmethod
    def from_param(cls: Type[_ndptr[None]], obj: ndarray[Any, Any]) -> _ctypes: ...
    @overload
    @classmethod
    def from_param(cls: Type[_ndptr[_DType]], obj: ndarray[Any, _DType]) -> _ctypes: ...

class _concrete_ndptr(_ndptr[_DType]):
    _dtype_: ClassVar[_DType]
    _shape_: ClassVar[Tuple[int, ...]]
    @property
    def contents(self) -> ndarray[Any, _DType]: ...

def load_library(
    libname: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    loader_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
) -> ctypes.CDLL: ...

__all__: List[str]
c_intp = _c_intp

@overload
def ndpointer(
    dtype: None = ...,
    ndim: int = ...,
    shape: None | _ShapeLike = ...,
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...,
) -> Type[_ndptr[None]]: ...
@overload
def ndpointer(
    dtype: _DTypeLike[_SCT],
    ndim: int = ...,
    *,
    shape: _ShapeLike,
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...
) -> Type[_concrete_ndptr[dtype[_SCT]]]: ...
@overload
def ndpointer(
    dtype: DTypeLike,
    ndim: int = ...,
    *,
    shape: _ShapeLike,
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...
) -> Type[_concrete_ndptr[dtype[Any]]]: ...
@overload
def ndpointer(
    dtype: _DTypeLike[_SCT],
    ndim: int = ...,
    shape: None = ...,
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...,
) -> Type[_ndptr[dtype[_SCT]]]: ...
@overload
def ndpointer(
    dtype: DTypeLike,
    ndim: int = ...,
    shape: None = ...,
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...,
) -> Type[_ndptr[dtype[Any]]]: ...
@overload
def as_ctypes_type(
    dtype: _BoolCodes | _DTypeLike[bool_] | Type[ctypes.c_bool],
) -> Type[ctypes.c_bool]: ...
@overload
def as_ctypes_type(
    dtype: _ByteCodes | _DTypeLike[byte] | Type[ctypes.c_byte],
) -> Type[ctypes.c_byte]: ...
@overload
def as_ctypes_type(
    dtype: _ShortCodes | _DTypeLike[short] | Type[ctypes.c_short],
) -> Type[ctypes.c_short]: ...
@overload
def as_ctypes_type(
    dtype: _IntCCodes | _DTypeLike[intc] | Type[ctypes.c_int],
) -> Type[ctypes.c_int]: ...
@overload
def as_ctypes_type(
    dtype: _IntCodes | _DTypeLike[int_] | Type[int | ctypes.c_long],
) -> Type[ctypes.c_long]: ...
@overload
def as_ctypes_type(
    dtype: _LongLongCodes | _DTypeLike[longlong] | Type[ctypes.c_longlong],
) -> Type[ctypes.c_longlong]: ...
@overload
def as_ctypes_type(
    dtype: _UByteCodes | _DTypeLike[ubyte] | Type[ctypes.c_ubyte],
) -> Type[ctypes.c_ubyte]: ...
@overload
def as_ctypes_type(
    dtype: _UShortCodes | _DTypeLike[ushort] | Type[ctypes.c_ushort],
) -> Type[ctypes.c_ushort]: ...
@overload
def as_ctypes_type(
    dtype: _UIntCCodes | _DTypeLike[uintc] | Type[ctypes.c_uint],
) -> Type[ctypes.c_uint]: ...
@overload
def as_ctypes_type(
    dtype: _UIntCodes | _DTypeLike[uint] | Type[ctypes.c_ulong],
) -> Type[ctypes.c_ulong]: ...
@overload
def as_ctypes_type(
    dtype: _ULongLongCodes | _DTypeLike[ulonglong] | Type[ctypes.c_ulonglong],
) -> Type[ctypes.c_ulonglong]: ...
@overload
def as_ctypes_type(
    dtype: _SingleCodes | _DTypeLike[single] | Type[ctypes.c_float],
) -> Type[ctypes.c_float]: ...
@overload
def as_ctypes_type(
    dtype: _DoubleCodes | _DTypeLike[double] | Type[float | ctypes.c_double],
) -> Type[ctypes.c_double]: ...
@overload
def as_ctypes_type(
    dtype: _LongDoubleCodes | _DTypeLike[longdouble] | Type[ctypes.c_longdouble],
) -> Type[ctypes.c_longdouble]: ...
@overload
def as_ctypes_type(dtype: _VoidDTypeLike) -> Type[Any]: ...
@overload
def as_ctypes_type(dtype: str) -> Type[Any]: ...
@overload
def as_array(obj: ctypes._PointerLike, shape: Sequence[int]) -> NDArray[Any]: ...
@overload
def as_array(
    obj: _ArrayLike[_SCT], shape: None | _ShapeLike = ...
) -> NDArray[_SCT]: ...
@overload
def as_array(obj: object, shape: None | _ShapeLike = ...) -> NDArray[Any]: ...
@overload
def as_ctypes(obj: bool_) -> ctypes.c_bool: ...
@overload
def as_ctypes(obj: byte) -> ctypes.c_byte: ...
@overload
def as_ctypes(obj: short) -> ctypes.c_short: ...
@overload
def as_ctypes(obj: intc) -> ctypes.c_int: ...
@overload
def as_ctypes(obj: int_) -> ctypes.c_long: ...
@overload
def as_ctypes(obj: longlong) -> ctypes.c_longlong: ...
@overload
def as_ctypes(obj: ubyte) -> ctypes.c_ubyte: ...
@overload
def as_ctypes(obj: ushort) -> ctypes.c_ushort: ...
@overload
def as_ctypes(obj: uintc) -> ctypes.c_uint: ...
@overload
def as_ctypes(obj: uint) -> ctypes.c_ulong: ...
@overload
def as_ctypes(obj: ulonglong) -> ctypes.c_ulonglong: ...
@overload
def as_ctypes(obj: single) -> ctypes.c_float: ...
@overload
def as_ctypes(obj: double) -> ctypes.c_double: ...
@overload
def as_ctypes(obj: longdouble) -> ctypes.c_longdouble: ...
@overload
def as_ctypes(obj: void) -> Any: ...
@overload
def as_ctypes(obj: NDArray[bool_]) -> ctypes.Array[ctypes.c_bool]: ...
@overload
def as_ctypes(obj: NDArray[byte]) -> ctypes.Array[ctypes.c_byte]: ...
@overload
def as_ctypes(obj: NDArray[short]) -> ctypes.Array[ctypes.c_short]: ...
@overload
def as_ctypes(obj: NDArray[intc]) -> ctypes.Array[ctypes.c_int]: ...
@overload
def as_ctypes(obj: NDArray[int_]) -> ctypes.Array[ctypes.c_long]: ...
@overload
def as_ctypes(obj: NDArray[longlong]) -> ctypes.Array[ctypes.c_longlong]: ...
@overload
def as_ctypes(obj: NDArray[ubyte]) -> ctypes.Array[ctypes.c_ubyte]: ...
@overload
def as_ctypes(obj: NDArray[ushort]) -> ctypes.Array[ctypes.c_ushort]: ...
@overload
def as_ctypes(obj: NDArray[uintc]) -> ctypes.Array[ctypes.c_uint]: ...
@overload
def as_ctypes(obj: NDArray[uint]) -> ctypes.Array[ctypes.c_ulong]: ...
@overload
def as_ctypes(obj: NDArray[ulonglong]) -> ctypes.Array[ctypes.c_ulonglong]: ...
@overload
def as_ctypes(obj: NDArray[single]) -> ctypes.Array[ctypes.c_float]: ...
@overload
def as_ctypes(obj: NDArray[double]) -> ctypes.Array[ctypes.c_double]: ...
@overload
def as_ctypes(obj: NDArray[longdouble]) -> ctypes.Array[ctypes.c_longdouble]: ...
@overload
def as_ctypes(obj: NDArray[void]) -> ctypes.Array[Any]: ...
