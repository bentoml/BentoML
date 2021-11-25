from typing import AnyStr

from pandas import DataFrame
from pandas._typing import FilePathOrBuffer, StorageOptions
from pandas.core import generic
from pandas.util._decorators import doc

""" feather-format compat """

@doc(storage_options=generic._shared_docs["storage_options"])
def to_feather(
    df: DataFrame,
    path: FilePathOrBuffer[AnyStr],
    storage_options: StorageOptions = ...,
    **kwargs
):  # -> None:
    """
    Write a DataFrame to the binary Feather format.

    Parameters
    ----------
    df : DataFrame
    path : string file path, or file-like object
    {storage_options}

        .. versionadded:: 1.2.0

    **kwargs :
        Additional keywords passed to `pyarrow.feather.write_feather`.

        .. versionadded:: 1.1.0
    """
    ...

@doc(storage_options=generic._shared_docs["storage_options"])
def read_feather(
    path, columns=..., use_threads: bool = ..., storage_options: StorageOptions = ...
):
    """
    Load a feather-format object from the file path.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.feather``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    columns : sequence, default None
        If not provided, all columns are read.
    use_threads : bool, default True
        Whether to parallelize reading using multiple threads.
    {storage_options}

        .. versionadded:: 1.2.0

    Returns
    -------
    type of object stored in file
    """
    ...
