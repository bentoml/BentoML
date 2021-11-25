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
    """
    Class for writing DataFrame objects into excel sheets.

    Default is to use xlwt for xls, openpyxl for xlsx, odf for ods.
    See DataFrame.to_excel for typical usage.

    The writer should be used as a context manager. Otherwise, call `close()` to save
    and close any opened file handles.

    Parameters
    ----------
    path : str or typing.BinaryIO
        Path to xls or xlsx or ods file.
    engine : str (optional)
        Engine to use for writing. If None, defaults to
        ``io.excel.<extension>.writer``.  NOTE: can only be passed as a keyword
        argument.

        .. deprecated:: 1.2.0

            As the `xlwt <https://pypi.org/project/xlwt/>`__ package is no longer
            maintained, the ``xlwt`` engine will be removed in a future
            version of pandas.

    date_format : str, default None
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
    datetime_format : str, default None
        Format string for datetime objects written into Excel files.
        (e.g. 'YYYY-MM-DD HH:MM:SS').
    mode : {'w', 'a'}, default 'w'
        File mode to use (write or append). Append does not work with fsspec URLs.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc., if using a URL that will
        be parsed by ``fsspec``, e.g., starting "s3://", "gcs://".

        .. versionadded:: 1.2.0
    if_sheet_exists : {'error', 'new', 'replace'}, default 'error'
        How to behave when trying to write to a sheet that already
        exists (append mode only).

        * error: raise a ValueError.
        * new: Create a new sheet, with a name determined by the engine.
        * replace: Delete the contents of the sheet before writing to it.

        .. versionadded:: 1.3.0
    engine_kwargs : dict, optional
        Keyword arguments to be passed into the engine.

        .. versionadded:: 1.3.0
    **kwargs : dict, optional
        Keyword arguments to be passed into the engine.

        .. deprecated:: 1.3.0

            Use engine_kwargs instead.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Notes
    -----
    None of the methods and properties are considered public.

    For compatibility with CSV writers, ExcelWriter serializes lists
    and dicts to strings before writing.

    Examples
    --------
    Default usage:

    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])
    >>> with ExcelWriter("path_to_file.xlsx") as writer:
    ...     df.to_excel(writer)

    To write to separate sheets in a single file:

    >>> df1 = pd.DataFrame([["AAA", "BBB"]], columns=["Spam", "Egg"])
    >>> df2 = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])
    >>> with ExcelWriter("path_to_file.xlsx") as writer:
    ...     df1.to_excel(writer, sheet_name="Sheet1")
    ...     df2.to_excel(writer, sheet_name="Sheet2")

    You can set the date format or datetime format:

    >>> from datetime import date, datetime
    >>> df = pd.DataFrame(
    ...     [
    ...         [date(2014, 1, 31), date(1999, 9, 24)],
    ...         [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)],
    ...     ],
    ...     index=["Date", "Datetime"],
    ...     columns=["X", "Y"],
    ... )
    >>> with ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     date_format="YYYY-MM-DD",
    ...     datetime_format="YYYY-MM-DD HH:MM:SS"
    ... ) as writer:
    ...     df.to_excel(writer)

    You can also append to an existing Excel file:

    >>> with ExcelWriter("path_to_file.xlsx", mode="a", engine="openpyxl") as writer:
    ...     df.to_excel(writer, sheet_name="Sheet3")

    You can store Excel file in RAM:

    >>> import io
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])
    >>> buffer = io.BytesIO()
    >>> with pd.ExcelWriter(buffer) as writer:
    ...     df.to_excel(writer)

    You can pack Excel file into zip archive:

    >>> import zipfile
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])
    >>> with zipfile.ZipFile("path_to_file.zip", "w") as zf:
    ...     with zf.open("filename.xlsx", "w") as buffer:
    ...         with pd.ExcelWriter(buffer) as writer:
    ...             df.to_excel(writer)
    """

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
    def supported_extensions(self):  # -> None:
        """Extensions that writer engine supports."""
        ...
    @property
    @abc.abstractmethod
    def engine(self):  # -> None:
        """Name of engine."""
        ...
    @abc.abstractmethod
    def write_cells(
        self, cells, sheet_name=..., startrow=..., startcol=..., freeze_panes=...
    ):  # -> None:
        """
        Write given formatted cells into Excel an excel sheet

        Parameters
        ----------
        cells : generator
            cell of formatted data to save to Excel sheet
        sheet_name : str, default None
            Name of Excel sheet, if None, then use self.cur_sheet
        startrow : upper left cell row to dump data frame
        startcol : upper left cell column to dump data frame
        freeze_panes: int tuple of length 2
            contains the bottom-most row and right-most column to freeze
        """
        ...
    @abc.abstractmethod
    def save(self):  # -> None:
        """
        Save workbook to disk.
        """
        ...
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
    def check_extension(cls, ext: str):  # -> Literal[True]:
        """
        checks that path's extension against the Writer's supported
        extensions.  If it isn't supported, raises UnsupportedFiletypeError.
        """
        ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...
    def close(self):  # -> None:
        """synonym for save, to make it more file-like"""
        ...

XLS_SIGNATURES = ...
ZIP_SIGNATURE = ...
PEEK_SIZE = ...

@doc(storage_options=_shared_docs["storage_options"])
def inspect_excel_format(
    content_or_path: FilePathOrBuffer, storage_options: StorageOptions = ...
) -> str | None:
    """
    Inspect the path or content of an excel file and get its format.

    Adopted from xlrd: https://github.com/python-excel/xlrd.

    Parameters
    ----------
    content_or_path : str or file-like object
        Path to file or content of file to inspect. May be a URL.
    {storage_options}

    Returns
    -------
    str or None
        Format of file if it can be determined.

    Raises
    ------
    ValueError
        If resulting stream is empty.
    BadZipFile
        If resulting stream does not have an XLS signature and is not a valid zipfile.
    """
    ...

class ExcelFile:
    """
    Class for parsing tabular excel sheets into DataFrame objects.

    See read_excel for more documentation.

    Parameters
    ----------
    path_or_buffer : str, path object (pathlib.Path or py._path.local.LocalPath),
        a file-like object, xlrd workbook or openpyxl workbook.
        If a string or path object, expected to be a path to a
        .xls, .xlsx, .xlsb, .xlsm, .odf, .ods, or .odt file.
    engine : str, default None
        If io is not a buffer or path, this must be set to identify io.
        Supported engines: ``xlrd``, ``openpyxl``, ``odf``, ``pyxlsb``
        Engine compatibility :

        - ``xlrd`` supports old-style Excel files (.xls).
        - ``openpyxl`` supports newer Excel file formats.
        - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
        - ``pyxlsb`` supports Binary Excel files.

        .. versionchanged:: 1.2.0

           The engine `xlrd <https://xlrd.readthedocs.io/en/latest/>`_
           now only supports old-style ``.xls`` files.
           When ``engine=None``, the following logic will be
           used to determine the engine:

           - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),
             then `odf <https://pypi.org/project/odfpy/>`_ will be used.
           - Otherwise if ``path_or_buffer`` is an xls format,
             ``xlrd`` will be used.
           - Otherwise if ``path_or_buffer`` is in xlsb format,
             `pyxlsb <https://pypi.org/project/pyxlsb/>`_ will be used.

           .. versionadded:: 1.3.0
           - Otherwise if `openpyxl <https://pypi.org/project/openpyxl/>`_ is installed,
             then ``openpyxl`` will be used.
           - Otherwise if ``xlrd >= 2.0`` is installed, a ``ValueError`` will be raised.
           - Otherwise ``xlrd`` will be used and a ``FutureWarning`` will be raised.
             This case will raise a ``ValueError`` in a future version of pandas.

           .. warning::

            Please do not report issues when using ``xlrd`` to read ``.xlsx`` files.
            This is not supported, switch to using ``openpyxl`` instead.
    """

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
    ):  # -> Any:
        """
        Parse specified sheet(s) into a DataFrame.

        Equivalent to read_excel(ExcelFile, ...)  See the read_excel
        docstring for more info on accepted parameters.

        Returns
        -------
        DataFrame or dict of DataFrames
            DataFrame from the passed in Excel file.
        """
        ...
    @property
    def book(self): ...
    @property
    def sheet_names(self): ...
    def close(self):  # -> None:
        """close io if necessary"""
        ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...
    def __del__(self): ...
