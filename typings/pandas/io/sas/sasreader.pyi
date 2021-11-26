from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Hashable, overload
from pandas import DataFrame
from pandas._typing import FilePathOrBuffer

if TYPE_CHECKING: ...

class ReaderBase(metaclass=ABCMeta):
    @abstractmethod
    def read(self, nrows=...): ...
    @abstractmethod
    def close(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

@overload
def read_sas(
    filepath_or_buffer: FilePathOrBuffer,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: int = ...,
    iterator: bool = ...,
) -> ReaderBase: ...
@overload
def read_sas(
    filepath_or_buffer: FilePathOrBuffer,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: None = ...,
    iterator: bool = ...,
) -> DataFrame | ReaderBase: ...
def read_sas(
    filepath_or_buffer: FilePathOrBuffer,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: int | None = ...,
    iterator: bool = ...,
) -> DataFrame | ReaderBase: ...
