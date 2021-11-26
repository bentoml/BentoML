from collections import abc
from typing import Any
from pandas._typing import DtypeArg, FilePathOrBuffer, StorageOptions
from pandas.core import generic
from pandas.util._decorators import Appender, deprecate_nonkeyword_arguments

_doc_read_csv_and_table = ...
_c_parser_defaults = ...
_fwf_defaults = ...
_c_unsupported = ...
_python_unsupported = ...
_deprecated_defaults: dict[str, Any] = ...
_deprecated_args: set[str] = ...

def validate_integer(name, val, min_val=...): ...
@deprecate_nonkeyword_arguments(
    version=None, allowed_args=["filepath_or_buffer"], stacklevel=3
)
@Appender(
    _doc_read_csv_and_table.format(
        func_name="read_csv",
        summary="Read a comma-separated values (csv) file into DataFrame.",
        _default_sep="','",
        storage_options=generic._shared_docs["storage_options"],
    )
)
def read_csv(
    filepath_or_buffer: FilePathOrBuffer,
    sep=...,
    delimiter=...,
    header=...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze=...,
    prefix=...,
    mangle_dupe_cols=...,
    dtype: DtypeArg | None = ...,
    engine=...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace=...,
    skiprows=...,
    skipfooter=...,
    nrows=...,
    na_values=...,
    keep_default_na=...,
    na_filter=...,
    verbose=...,
    skip_blank_lines=...,
    parse_dates=...,
    infer_datetime_format=...,
    keep_date_col=...,
    date_parser=...,
    dayfirst=...,
    cache_dates=...,
    iterator=...,
    chunksize=...,
    compression=...,
    thousands=...,
    decimal: str = ...,
    lineterminator=...,
    quotechar=...,
    quoting=...,
    doublequote=...,
    escapechar=...,
    comment=...,
    encoding=...,
    encoding_errors: str | None = ...,
    dialect=...,
    error_bad_lines=...,
    warn_bad_lines=...,
    on_bad_lines=...,
    delim_whitespace=...,
    low_memory=...,
    memory_map=...,
    float_precision=...,
    storage_options: StorageOptions = ...,
): ...
@deprecate_nonkeyword_arguments(
    version=None, allowed_args=["filepath_or_buffer"], stacklevel=3
)
@Appender(
    _doc_read_csv_and_table.format(
        func_name="read_table",
        summary="Read general delimited file into DataFrame.",
        _default_sep=r"'\\t' (tab-stop)",
        storage_options=generic._shared_docs["storage_options"],
    )
)
def read_table(
    filepath_or_buffer: FilePathOrBuffer,
    sep=...,
    delimiter=...,
    header=...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze=...,
    prefix=...,
    mangle_dupe_cols=...,
    dtype: DtypeArg | None = ...,
    engine=...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace=...,
    skiprows=...,
    skipfooter=...,
    nrows=...,
    na_values=...,
    keep_default_na=...,
    na_filter=...,
    verbose=...,
    skip_blank_lines=...,
    parse_dates=...,
    infer_datetime_format=...,
    keep_date_col=...,
    date_parser=...,
    dayfirst=...,
    cache_dates=...,
    iterator=...,
    chunksize=...,
    compression=...,
    thousands=...,
    decimal: str = ...,
    lineterminator=...,
    quotechar=...,
    quoting=...,
    doublequote=...,
    escapechar=...,
    comment=...,
    encoding=...,
    dialect=...,
    error_bad_lines=...,
    warn_bad_lines=...,
    on_bad_lines=...,
    encoding_errors: str | None = ...,
    delim_whitespace=...,
    low_memory=...,
    memory_map=...,
    float_precision=...,
): ...
def read_fwf(
    filepath_or_buffer: FilePathOrBuffer,
    colspecs=...,
    widths=...,
    infer_nrows=...,
    **kwds
): ...

class TextFileReader(abc.Iterator):
    def __init__(self, f, engine=..., **kwds) -> None: ...
    def close(self): ...
    def __next__(self): ...
    def read(self, nrows=...): ...
    def get_chunk(self, size=...): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

def TextParser(*args, **kwds): ...

MANDATORY_DIALECT_ATTRS = ...
