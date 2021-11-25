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
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        """Object to write in JSON format."""
        ...

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
    ) -> None:
        """
        Adds a `schema` attribute with the Table Schema, resets
        the index (can't do in caller, because the schema inference needs
        to know what the index is, forces orient to records, and forces
        date_format to 'iso'.
        """
        ...
    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]: ...

@doc(storage_options=generic._shared_docs["storage_options"])
@deprecate_kwarg(old_arg_name="numpy", new_arg_name=None)
@deprecate_nonkeyword_arguments(version="2.0", allowed_args=["path_or_buf"], stacklevel=3)
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
):  # -> JsonReader | DataFrame:
    """
    Convert a JSON string to pandas object.

    Parameters
    ----------
    path_or_buf : a valid JSON str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.json``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    orient : str
        Indication of expected JSON string format.
        Compatible JSON strings can be produced by ``to_json()`` with a
        corresponding orient value.
        The set of possible orients is:

        - ``'split'`` : dict like
          ``{{index -> [index], columns -> [columns], data -> [values]}}``
        - ``'records'`` : list like
          ``[{{column -> value}}, ... , {{column -> value}}]``
        - ``'index'`` : dict like ``{{index -> {{column -> value}}}}``
        - ``'columns'`` : dict like ``{{column -> {{index -> value}}}}``
        - ``'values'`` : just the values array

        The allowed and default values depend on the value
        of the `typ` parameter.

        * when ``typ == 'series'``,

          - allowed orients are ``{{'split','records','index'}}``
          - default is ``'index'``
          - The Series index must be unique for orient ``'index'``.

        * when ``typ == 'frame'``,

          - allowed orients are ``{{'split','records','index',
            'columns','values', 'table'}}``
          - default is ``'columns'``
          - The DataFrame index must be unique for orients ``'index'`` and
            ``'columns'``.
          - The DataFrame columns must be unique for orients ``'index'``,
            ``'columns'``, and ``'records'``.

    typ : {{'frame', 'series'}}, default 'frame'
        The type of object to recover.

    dtype : bool or dict, default None
        If True, infer dtypes; if a dict of column to dtype, then use those;
        if False, then don't infer dtypes at all, applies only to the data.

        For all ``orient`` values except ``'table'``, default is True.

        .. versionchanged:: 0.25.0

           Not applicable for ``orient='table'``.

    convert_axes : bool, default None
        Try to convert the axes to the proper dtypes.

        For all ``orient`` values except ``'table'``, default is True.

        .. versionchanged:: 0.25.0

           Not applicable for ``orient='table'``.

    convert_dates : bool or list of str, default True
        If True then default datelike columns may be converted (depending on
        keep_default_dates).
        If False, no dates will be converted.
        If a list of column names, then those columns will be converted and
        default datelike columns may also be converted (depending on
        keep_default_dates).

    keep_default_dates : bool, default True
        If parsing dates (convert_dates is not False), then try to parse the
        default datelike columns.
        A column label is datelike if

        * it ends with ``'_at'``,

        * it ends with ``'_time'``,

        * it begins with ``'timestamp'``,

        * it is ``'modified'``, or

        * it is ``'date'``.

    numpy : bool, default False
        Direct decoding to numpy arrays. Supports numeric data only, but
        non-numeric column and index labels are supported. Note also that the
        JSON ordering MUST be the same for each term if numpy=True.

        .. deprecated:: 1.0.0

    precise_float : bool, default False
        Set to enable usage of higher precision (strtod) function when
        decoding string to double values. Default (False) is to use fast but
        less precise builtin functionality.

    date_unit : str, default None
        The timestamp unit to detect if converting dates. The default behaviour
        is to try and detect the correct precision, but if this is not desired
        then pass one of 's', 'ms', 'us' or 'ns' to force parsing only seconds,
        milliseconds, microseconds or nanoseconds respectively.

    encoding : str, default is 'utf-8'
        The encoding to use to decode py3 bytes.

    encoding_errors : str, optional, default "strict"
        How encoding errors are treated. `List of possible values
        <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .

        .. versionadded:: 1.3.0

    lines : bool, default False
        Read the file as a json object per line.

    chunksize : int, optional
        Return JsonReader object for iteration.
        See the `line-delimited json docs
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json>`_
        for more information on ``chunksize``.
        This can only be passed if `lines=True`.
        If this is None, the file will be read into memory all at once.

        .. versionchanged:: 1.2

           ``JsonReader`` is a context manager.

    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default 'infer'
        For on-the-fly decompression of on-disk data. If 'infer', then use
        gzip, bz2, zip or xz if path_or_buf is a string ending in
        '.gz', '.bz2', '.zip', or 'xz', respectively, and no decompression
        otherwise. If using 'zip', the ZIP file must contain only one data
        file to be read in. Set to None for no decompression.

    nrows : int, optional
        The number of lines from the line-delimited jsonfile that has to be read.
        This can only be passed if `lines=True`.
        If this is None, all the rows will be returned.

        .. versionadded:: 1.1

    {storage_options}

        .. versionadded:: 1.2.0

    Returns
    -------
    Series or DataFrame
        The type returned depends on the value of `typ`.

    See Also
    --------
    DataFrame.to_json : Convert a DataFrame to a JSON string.
    Series.to_json : Convert a Series to a JSON string.

    Notes
    -----
    Specific to ``orient='table'``, if a :class:`DataFrame` with a literal
    :class:`Index` name of `index` gets written with :func:`to_json`, the
    subsequent read operation will incorrectly set the :class:`Index` name to
    ``None``. This is because `index` is also used by :func:`DataFrame.to_json`
    to denote a missing :class:`Index` name, and the subsequent
    :func:`read_json` operation cannot distinguish between the two. The same
    limitation is encountered with a :class:`MultiIndex` and any names
    beginning with ``'level_'``.

    Examples
    --------
    >>> df = pd.DataFrame([['a', 'b'], ['c', 'd']],
    ...                   index=['row 1', 'row 2'],
    ...                   columns=['col 1', 'col 2'])

    Encoding/decoding a Dataframe using ``'split'`` formatted JSON:

    >>> df.to_json(orient='split')
        '\
{{\
"columns":["col 1","col 2"],\
"index":["row 1","row 2"],\
"data":[["a","b"],["c","d"]]\
}}\
'
    >>> pd.read_json(_, orient='split')
          col 1 col 2
    row 1     a     b
    row 2     c     d

    Encoding/decoding a Dataframe using ``'index'`` formatted JSON:

    >>> df.to_json(orient='index')
    '{{"row 1":{{"col 1":"a","col 2":"b"}},"row 2":{{"col 1":"c","col 2":"d"}}}}'

    >>> pd.read_json(_, orient='index')
          col 1 col 2
    row 1     a     b
    row 2     c     d

    Encoding/decoding a Dataframe using ``'records'`` formatted JSON.
    Note that index labels are not preserved with this encoding.

    >>> df.to_json(orient='records')
    '[{{"col 1":"a","col 2":"b"}},{{"col 1":"c","col 2":"d"}}]'
    >>> pd.read_json(_, orient='records')
      col 1 col 2
    0     a     b
    1     c     d

    Encoding with Table Schema

    >>> df.to_json(orient='table')
        '\
{{"schema":{{"fields":[\
{{"name":"index","type":"string"}},\
{{"name":"col 1","type":"string"}},\
{{"name":"col 2","type":"string"}}],\
"primaryKey":["index"],\
"pandas_version":"0.20.0"}},\
"data":[\
{{"index":"row 1","col 1":"a","col 2":"b"}},\
{{"index":"row 2","col 1":"c","col 2":"d"}}]\
}}\
'
    """
    ...

class JsonReader(abc.Iterator):
    """
    JsonReader provides an interface for reading in a JSON file.

    If initialized with ``lines=True`` and ``chunksize``, can be iterated over
    ``chunksize`` lines at a time. Otherwise, calling ``read`` reads in the
    whole document.
    """

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
    def read(self):  # -> DataFrame:
        """
        Read the whole JSON input into a pandas object.
        """
        ...
    def close(self):  # -> None:
        """
        If we opened a stream earlier, in _get_data_from_filepath, we should
        close it.

        If an open stream or file was passed, we leave it open.
        """
        ...
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
    def check_keys_split(self, decoded):  # -> None:
        """
        Checks that dict has only the appropriate keys for orient='split'.
        """
        ...
    def parse(self): ...

class SeriesParser(Parser):
    _default_orient = ...
    _split_keys = ...

class FrameParser(Parser):
    _default_orient = ...
    _split_keys = ...
