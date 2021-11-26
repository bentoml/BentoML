from abc import ABC, abstractmethod
from collections import abc
from typing import Any, Callable, Mapping
from pandas._typing import (
    CompressionOptions,
    DtypeArg,
    IndexLabel,
    JSONSerializable,
    StorageOptions,
)
from pandas.core import generic
from pandas.core.generic import NDFrame
from pandas.util._decorators import deprecate_kwarg, deprecate_nonkeyword_arguments, doc

loads = ...
dumps = ...
TABLE_SCHEMA_VERSION = ...

def to_json(
    path_or_buf,
    obj: NDFrame,
    orient: str | None = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
): ...

class Writer(ABC):
    _default_orient: str
    def __init__(
        self,
        obj,
        orient: str | None,
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        indent: int = ...,
    ) -> None: ...
    def write(self): ...
    @property
    @abstractmethod
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]: ...

class SeriesWriter(Writer):
    _default_orient = ...
    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]: ...

class FrameWriter(Writer):
    _default_orient = ...
    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]: ...

class JSONTableWriter(FrameWriter):
    _default_orient = ...
    def __init__(
        self,
        obj,
        orient: str | None,
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        indent: int = ...,
    ) -> None: ...
    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]: ...

@doc(storage_options=generic._shared_docs["storage_options"])
@deprecate_kwarg(old_arg_name="numpy", new_arg_name=None)
@deprecate_nonkeyword_arguments(
    version="2.0", allowed_args=["path_or_buf"], stacklevel=3
)
def read_json(
    path_or_buf=...,
    orient=...,
    typ=...,
    dtype: DtypeArg | None = ...,
    convert_axes=...,
    convert_dates=...,
    keep_default_dates: bool = ...,
    numpy: bool = ...,
    precise_float: bool = ...,
    date_unit=...,
    encoding=...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
): ...

class JsonReader(abc.Iterator):
    def __init__(
        self,
        filepath_or_buffer,
        orient,
        typ,
        dtype,
        convert_axes,
        convert_dates,
        keep_default_dates: bool,
        numpy: bool,
        precise_float: bool,
        date_unit,
        encoding,
        lines: bool,
        chunksize: int | None,
        compression: CompressionOptions,
        nrows: int | None,
        storage_options: StorageOptions = ...,
        encoding_errors: str | None = ...,
    ) -> None: ...
    def read(self): ...
    def close(self): ...
    def __next__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class Parser:
    _split_keys: tuple[str, ...]
    _default_orient: str
    _STAMP_UNITS = ...
    _MIN_STAMPS = ...
    def __init__(
        self,
        json,
        orient,
        dtype: DtypeArg | None = ...,
        convert_axes=...,
        convert_dates=...,
        keep_default_dates=...,
        numpy=...,
        precise_float=...,
        date_unit=...,
    ) -> None: ...
    def check_keys_split(self, decoded): ...
    def parse(self): ...

class SeriesParser(Parser):
    _default_orient = ...
    _split_keys = ...

class FrameParser(Parser):
    _default_orient = ...
    _split_keys = ...
