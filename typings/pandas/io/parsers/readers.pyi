from collections import abc
from typing import Any

from pandas._typing import DtypeArg, FilePathOrBuffer, StorageOptions
from pandas.core import generic
from pandas.util._decorators import Appender, deprecate_nonkeyword_arguments

"""
Module contains tools for processing files into DataFrames or other objects
"""
_doc_read_csv_and_table = ...
_c_parser_defaults = ...
_fwf_defaults = ...
_c_unsupported = ...
_python_unsupported = ...
_deprecated_defaults: dict[str, Any] = ...
_deprecated_args: set[str] = ...

def validate_integer(name, val, min_val=...):  # -> int:
    """
    Checks whether the 'name' parameter for parsing is either
    an integer OR float that can SAFELY be cast to an integer
    without losing accuracy. Raises a ValueError if that is
    not the case.

    Parameters
    ----------
    name : str
        Parameter name (used for error reporting)
    val : int or float
        The value to check
    min_val : int
        Minimum allowed value (val < min_val will result in a ValueError)
    """
    ...

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
):  # -> TextFileReader | DataFrame:
    r"""
    Read a table of fixed-width formatted lines into DataFrame.

    Also supports optionally iterating or breaking of the file
    into chunks.

    Additional help can be found in the `online docs for IO Tools
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.csv``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    colspecs : list of tuple (int, int) or 'infer'. optional
        A list of tuples giving the extents of the fixed-width
        fields of each line as half-open intervals (i.e.,  [from, to[ ).
        String value 'infer' can be used to instruct the parser to try
        detecting the column specifications from the first 100 rows of
        the data which are not being skipped via skiprows (default='infer').
    widths : list of int, optional
        A list of field widths which can be used instead of 'colspecs' if
        the intervals are contiguous.
    infer_nrows : int, default 100
        The number of rows to consider when letting the parser determine the
        `colspecs`.
    **kwds : optional
        Optional keyword arguments can be passed to ``TextFileReader``.

    Returns
    -------
    DataFrame or TextParser
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.

    See Also
    --------
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_csv : Read a comma-separated values (csv) file into DataFrame.

    Examples
    --------
    >>> pd.read_fwf('data.csv')  # doctest: +SKIP
    """
    ...

class TextFileReader(abc.Iterator):
    """

    Passed dialect overrides any of the related parser options

    """

    def __init__(self, f, engine=..., **kwds) -> None: ...
    def close(self): ...
    def __next__(self): ...
    def read(self, nrows=...): ...
    def get_chunk(self, size=...): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

def TextParser(*args, **kwds):  # -> TextFileReader:
    """
    Converts lists of lists/tuples into DataFrames with proper type inference
    and optional (e.g. string to datetime) conversion. Also enables iterating
    lazily over chunks of large files

    Parameters
    ----------
    data : file-like object or list
    delimiter : separator character to use
    dialect : str or csv.Dialect instance, optional
        Ignored if delimiter is longer than 1 character
    names : sequence, default
    header : int, default 0
        Row to use to parse column labels. Defaults to the first row. Prior
        rows will be discarded
    index_col : int or list, optional
        Column or columns to use as the (possibly hierarchical) index
    has_index_names: bool, default False
        True if the cols defined in index_col have an index name and are
        not in the header.
    na_values : scalar, str, list-like, or dict, optional
        Additional strings to recognize as NA/NaN.
    keep_default_na : bool, default True
    thousands : str, optional
        Thousands separator
    comment : str, optional
        Comment out remainder of line
    parse_dates : bool, default False
    keep_date_col : bool, default False
    date_parser : function, optional
    skiprows : list of integers
        Row numbers to skip
    skipfooter : int
        Number of line at bottom of file to skip
    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the cell (not column) content, and return the
        transformed content.
    encoding : str, optional
        Encoding to use for UTF when reading/writing (ex. 'utf-8')
    squeeze : bool, default False
        returns Series if only one column.
    infer_datetime_format: bool, default False
        If True and `parse_dates` is True for a column, try to infer the
        datetime format based on the first datetime string. If the format
        can be inferred, there often will be a large parsing speed-up.
    float_precision : str, optional
        Specifies which converter the C engine should use for floating-point
        values. The options are `None` or `high` for the ordinary converter,
        `legacy` for the original lower precision pandas converter, and
        `round_trip` for the round-trip converter.

        .. versionchanged:: 1.2
    """
    ...

MANDATORY_DIALECT_ATTRS = ...
