from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Hashable, overload

from pandas import DataFrame
from pandas._typing import FilePathOrBuffer

"""
Read SAS sas7bdat or xport files.
"""
if TYPE_CHECKING: ...

class ReaderBase(metaclass=ABCMeta):
    """
    Protocol for XportReader and SAS7BDATReader classes.
    """

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
) -> DataFrame | ReaderBase:
    """
    Read SAS files stored as either XPORT or SAS7BDAT format files.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.sas``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    format : str {'xport', 'sas7bdat'} or None
        If None, file format is inferred from file extension. If 'xport' or
        'sas7bdat', uses the corresponding format.
    index : identifier of index column, defaults to None
        Identifier of column that should be used as index of the DataFrame.
    encoding : str, default is None
        Encoding for text data.  If None, text data are stored as raw bytes.
    chunksize : int
        Read file `chunksize` lines at a time, returns iterator.

        .. versionchanged:: 1.2

            ``TextFileReader`` is a context manager.
    iterator : bool, defaults to False
        If True, returns an iterator for reading the file incrementally.

        .. versionchanged:: 1.2

            ``TextFileReader`` is a context manager.

    Returns
    -------
    DataFrame if iterator=False and chunksize=None, else SAS7BDATReader
    or XportReader
    """
    ...
