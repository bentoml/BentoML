import os
import sys
import types
import zipfile
from typing import IO, Any, Callable, Generic, Iterable, Iterator, List
from typing import Literal as L
from typing import Mapping, Pattern, Protocol, Sequence, Type, TypeVar, Union, overload
from numpy import DataSource as DataSource
from numpy import dtype, float64, generic, ndarray, recarray, record, void
from numpy.core.multiarray import packbits as packbits
from numpy.core.multiarray import unpackbits as unpackbits
from numpy.ma.mrecords import MaskedRecords
from numpy.typing import ArrayLike, DTypeLike, NDArray, _SupportsDType

_T = TypeVar("_T")
_T_contra = TypeVar("_T_contra", contravariant=True)
_T_co = TypeVar("_T_co", covariant=True)
_SCT = TypeVar("_SCT", bound=generic)
_CharType_co = TypeVar("_CharType_co", str, bytes, covariant=True)
_CharType_contra = TypeVar("_CharType_contra", str, bytes, contravariant=True)
_DTypeLike = Union[Type[_SCT], dtype[_SCT], _SupportsDType[dtype[_SCT]]]

class _SupportsGetItem(Protocol[_T_contra, _T_co]):
    def __getitem__(self, key: _T_contra, /) -> _T_co: ...

class _SupportsRead(Protocol[_CharType_co]):
    def read(self) -> _CharType_co: ...

class _SupportsReadSeek(Protocol[_CharType_co]):
    def read(self, n: int, /) -> _CharType_co: ...
    def seek(self, offset: int, whence: int, /) -> object: ...

class _SupportsWrite(Protocol[_CharType_contra]):
    def write(self, s: _CharType_contra, /) -> object: ...

__all__: List[str]

class BagObj(Generic[_T_co]):
    def __init__(self, obj: _SupportsGetItem[str, _T_co]) -> None: ...
    def __getattribute__(self, key: str) -> _T_co: ...
    def __dir__(self) -> List[str]: ...

class NpzFile(Mapping[str, NDArray[Any]]):
    zip: zipfile.ZipFile
    fid: None | IO[str]
    files: List[str]
    allow_pickle: bool
    pickle_kwargs: None | Mapping[str, Any]
    @property
    def f(self: _T) -> BagObj[_T]: ...
    @f.setter
    def f(self: _T, value: BagObj[_T]) -> None: ...
    def __init__(
        self,
        fid: IO[str],
        own_fid: bool = ...,
        allow_pickle: bool = ...,
        pickle_kwargs: None | Mapping[str, Any] = ...,
    ) -> None: ...
    def __enter__(self: _T) -> _T: ...
    def __exit__(
        self,
        exc_type: None | Type[BaseException],
        exc_value: None | BaseException,
        traceback: None | types.TracebackType,
        /,
    ) -> None: ...
    def close(self) -> None: ...
    def __del__(self) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: str) -> NDArray[Any]: ...

def load(
    file: str | bytes | os.PathLike[Any] | _SupportsReadSeek[bytes],
    mmap_mode: L[None, "r+", "r", "w+", "c"] = ...,
    allow_pickle: bool = ...,
    fix_imports: bool = ...,
    encoding: L["ASCII", "latin1", "bytes"] = ...,
) -> Any: ...
def save(
    file: str | os.PathLike[str] | _SupportsWrite[bytes],
    arr: ArrayLike,
    allow_pickle: bool = ...,
    fix_imports: bool = ...,
) -> None: ...
def savez(
    file: str | os.PathLike[str] | _SupportsWrite[bytes],
    *args: ArrayLike,
    **kwds: ArrayLike,
) -> None: ...
def savez_compressed(
    file: str | os.PathLike[str] | _SupportsWrite[bytes],
    *args: ArrayLike,
    **kwds: ArrayLike,
) -> None: ...
@overload
def loadtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: None = ...,
    comments: str | Sequence[str] = ...,
    delimiter: None | str = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    skiprows: int = ...,
    usecols: int | Sequence[int] = ...,
    unpack: bool = ...,
    ndmin: L[0, 1, 2] = ...,
    encoding: None | str = ...,
    max_rows: None | int = ...,
    *,
    like: None | ArrayLike = ...,
) -> NDArray[float64]: ...
@overload
def loadtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: _DTypeLike[_SCT],
    comments: str | Sequence[str] = ...,
    delimiter: None | str = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    skiprows: int = ...,
    usecols: int | Sequence[int] = ...,
    unpack: bool = ...,
    ndmin: L[0, 1, 2] = ...,
    encoding: None | str = ...,
    max_rows: None | int = ...,
    *,
    like: None | ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def loadtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: DTypeLike,
    comments: str | Sequence[str] = ...,
    delimiter: None | str = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    skiprows: int = ...,
    usecols: int | Sequence[int] = ...,
    unpack: bool = ...,
    ndmin: L[0, 1, 2] = ...,
    encoding: None | str = ...,
    max_rows: None | int = ...,
    *,
    like: None | ArrayLike = ...,
) -> NDArray[Any]: ...
def savetxt(
    fname: str | os.PathLike[str] | _SupportsWrite[str] | _SupportsWrite[bytes],
    X: ArrayLike,
    fmt: str | Sequence[str] = ...,
    delimiter: str = ...,
    newline: str = ...,
    header: str = ...,
    footer: str = ...,
    comments: str = ...,
    encoding: None | str = ...,
) -> None: ...
@overload
def fromregex(
    file: str | os.PathLike[str] | _SupportsRead[str] | _SupportsRead[bytes],
    regexp: str | bytes | Pattern[Any],
    dtype: _DTypeLike[_SCT],
    encoding: None | str = ...,
) -> NDArray[_SCT]: ...
@overload
def fromregex(
    file: str | os.PathLike[str] | _SupportsRead[str] | _SupportsRead[bytes],
    regexp: str | bytes | Pattern[Any],
    dtype: DTypeLike,
    encoding: None | str = ...,
) -> NDArray[Any]: ...
@overload
def genfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: None = ...,
    *args: Any,
    **kwargs: Any,
) -> NDArray[float64]: ...
@overload
def genfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: _DTypeLike[_SCT],
    *args: Any,
    **kwargs: Any,
) -> NDArray[_SCT]: ...
@overload
def genfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: DTypeLike,
    *args: Any,
    **kwargs: Any,
) -> NDArray[Any]: ...
@overload
def recfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[False] = ...,
    **kwargs: Any,
) -> recarray[Any, dtype[record]]: ...
@overload
def recfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[True],
    **kwargs: Any,
) -> MaskedRecords[Any, dtype[void]]: ...
@overload
def recfromcsv(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[False] = ...,
    **kwargs: Any,
) -> recarray[Any, dtype[record]]: ...
@overload
def recfromcsv(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[True],
    **kwargs: Any,
) -> MaskedRecords[Any, dtype[void]]: ...
