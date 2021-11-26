import abc
from typing import Any, Mapping
from pandas._typing import DtypeArg, FilePathOrBuffer, StorageOptions
from pandas.core.shared_docs import _shared_docs
from pandas.util._decorators import Appender, deprecate_nonkeyword_arguments, doc

_read_excel_doc = ...

@deprecate_nonkeyword_arguments(allowed_args=["io", "sheet_name"], version="2.0")
@Appender(_read_excel_doc)
def read_excel(
    io,
    sheet_name=...,
    header=...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze=...,
    dtype: DtypeArg | None = ...,
    engine=...,
    converters=...,
    true_values=...,
    false_values=...,
    skiprows=...,
    nrows=...,
    na_values=...,
    keep_default_na=...,
    na_filter=...,
    verbose=...,
    parse_dates=...,
    date_parser=...,
    thousands=...,
    comment=...,
    skipfooter=...,
    convert_float=...,
    mangle_dupe_cols=...,
    storage_options: StorageOptions = ...,
): ...

class BaseExcelReader(metaclass=abc.ABCMeta):
    def __init__(
        self, filepath_or_buffer, storage_options: StorageOptions = ...
    ) -> None: ...
    @abc.abstractmethod
    def load_workbook(self, filepath_or_buffer): ...
    def close(self): ...
    @property
    @abc.abstractmethod
    def sheet_names(self): ...
    @abc.abstractmethod
    def get_sheet_by_name(self, name): ...
    @abc.abstractmethod
    def get_sheet_by_index(self, index): ...
    @abc.abstractmethod
    def get_sheet_data(self, sheet, convert_float): ...
    def raise_if_bad_sheet_by_index(self, index: int) -> None: ...
    def raise_if_bad_sheet_by_name(self, name: str) -> None: ...
    def parse(
        self,
        sheet_name=...,
        header=...,
        names=...,
        index_col=...,
        usecols=...,
        squeeze=...,
        dtype: DtypeArg | None = ...,
        true_values=...,
        false_values=...,
        skiprows=...,
        nrows=...,
        na_values=...,
        verbose=...,
        parse_dates=...,
        date_parser=...,
        thousands=...,
        comment=...,
        skipfooter=...,
        convert_float=...,
        mangle_dupe_cols=...,
        **kwds
    ): ...

class ExcelWriter(metaclass=abc.ABCMeta):
    def __new__(
        cls,
        path: FilePathOrBuffer | ExcelWriter,
        engine=...,
        date_format=...,
        datetime_format=...,
        mode: str = ...,
        storage_options: StorageOptions = ...,
        if_sheet_exists: str | None = ...,
        engine_kwargs: dict | None = ...,
        **kwargs
    ): ...
    path = ...
    @property
    @abc.abstractmethod
    def supported_extensions(self): ...
    @property
    @abc.abstractmethod
    def engine(self): ...
    @abc.abstractmethod
    def write_cells(
        self, cells, sheet_name=..., startrow=..., startcol=..., freeze_panes=...
    ): ...
    @abc.abstractmethod
    def save(self): ...
    def __init__(
        self,
        path: FilePathOrBuffer | ExcelWriter,
        engine=...,
        date_format=...,
        datetime_format=...,
        mode: str = ...,
        storage_options: StorageOptions = ...,
        if_sheet_exists: str | None = ...,
        engine_kwargs: dict | None = ...,
        **kwargs
    ) -> None: ...
    def __fspath__(self): ...
    @classmethod
    def check_extension(cls, ext: str): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...
    def close(self): ...

XLS_SIGNATURES = ...
ZIP_SIGNATURE = ...
PEEK_SIZE = ...

@doc(storage_options=_shared_docs["storage_options"])
def inspect_excel_format(
    content_or_path: FilePathOrBuffer, storage_options: StorageOptions = ...
) -> str | None: ...

class ExcelFile:
    _engines: Mapping[str, Any] = ...
    def __init__(
        self, path_or_buffer, engine=..., storage_options: StorageOptions = ...
    ) -> None: ...
    def __fspath__(self): ...
    def parse(
        self,
        sheet_name=...,
        header=...,
        names=...,
        index_col=...,
        usecols=...,
        squeeze=...,
        converters=...,
        true_values=...,
        false_values=...,
        skiprows=...,
        nrows=...,
        na_values=...,
        parse_dates=...,
        date_parser=...,
        thousands=...,
        comment=...,
        skipfooter=...,
        convert_float=...,
        mangle_dupe_cols=...,
        **kwds
    ): ...
    @property
    def book(self): ...
    @property
    def sheet_names(self): ...
    def close(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...
