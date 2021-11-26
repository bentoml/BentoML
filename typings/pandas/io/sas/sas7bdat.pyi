from collections import abc
import numpy as np
from pandas import DataFrame
from pandas.io.sas.sasreader import ReaderBase

class _SubheaderPointer:
    offset: int
    length: int
    compression: int
    ptype: int
    def __init__(
        self, offset: int, length: int, compression: int, ptype: int
    ) -> None: ...

class _Column:
    col_id: int
    name: str | bytes
    label: str | bytes
    format: str | bytes
    ctype: bytes
    length: int
    def __init__(
        self,
        col_id: int,
        name: str | bytes,
        label: str | bytes,
        format: str | bytes,
        ctype: bytes,
        length: int,
    ) -> None: ...

class SAS7BDATReader(ReaderBase, abc.Iterator):
    _int_length: int
    _cached_page: bytes | None
    def __init__(
        self,
        path_or_buf,
        index=...,
        convert_dates=...,
        blank_missing=...,
        chunksize=...,
        encoding=...,
        convert_text=...,
        convert_header_text=...,
    ) -> None: ...
    def column_data_lengths(self) -> np.ndarray: ...
    def column_data_offsets(self) -> np.ndarray: ...
    def column_types(self) -> np.ndarray: ...
    def close(self) -> None: ...
    def __next__(self): ...
    def read(self, nrows: int | None = ...) -> DataFrame | None: ...
