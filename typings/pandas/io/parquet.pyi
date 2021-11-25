from typing import AnyStr

from pandas import DataFrame
from pandas._typing import FilePathOrBuffer, StorageOptions
from pandas.core import generic
from pandas.util._decorators import doc

""" parquet compat """

def get_engine(engine: str) -> BaseImpl:
    """return our implementation"""
    ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame): ...
    def write(self, df: DataFrame, path, compression, **kwargs): ...
    def read(self, path, columns=..., **kwargs): ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePathOrBuffer[AnyStr],
        compression: str | None = ...,
        index: bool | None = ...,
        storage_options: StorageOptions = ...,
        partition_cols: list[str] | None = ...,
        **kwargs
    ): ...
    def read(
        self,
        path,
        columns=...,
        use_nullable_dtypes=...,
        storage_options: StorageOptions = ...,
        **kwargs
    ): ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path,
        compression=...,
        index=...,
        partition_cols=...,
        storage_options: StorageOptions = ...,
        **kwargs
    ): ...
    def read(
        self, path, columns=..., storage_options: StorageOptions = ..., **kwargs
    ): ...

@doc(storage_options=generic._shared_docs["storage_options"])
def to_parquet(
    df: DataFrame,
    path: FilePathOrBuffer | None = ...,
    engine: str = ...,
    compression: str | None = ...,
    index: bool | None = ...,
    storage_options: StorageOptions = ...,
    partition_cols: list[str] | None = ...,
    **kwargs
) -> bytes | None:
    """
    Write a DataFrame to the parquet format.

    Parameters
    ----------
    df : DataFrame
    path : str or file-like object, default None
        If a string, it will be used as Root Directory path
        when writing a partitioned dataset. By file-like object,
        we refer to objects with a write() method, such as a file handle
        (e.g. via builtin open function) or io.BytesIO. The engine
        fastparquet does not accept file-like objects. If path is None,
        a bytes object is returned.

        .. versionchanged:: 1.2.0

    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.
    compression : {{'snappy', 'gzip', 'brotli', None}}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file.
        If ``None``, similar to ``True`` the dataframe's index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn't require much space and is faster. Other indexes will
        be included as columns in the file output.
    partition_cols : str or list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.
    {storage_options}

        .. versionadded:: 1.2.0

    kwargs
        Additional keyword arguments passed to the engine

    Returns
    -------
    bytes if no path argument is provided else None
    """
    ...

@doc(storage_options=generic._shared_docs["storage_options"])
def read_parquet(
    path,
    engine: str = ...,
    columns=...,
    storage_options: StorageOptions = ...,
    use_nullable_dtypes: bool = ...,
    **kwargs
):  # -> NoReturn:
    """
    Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables`` or ``s3://bucket/partition_dir``

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.
    columns : list, default=None
        If not None, only these columns will be read from the file.

    {storage_options}

        .. versionadded:: 1.3.0

    use_nullable_dtypes : bool, default False
        If True, use dtypes that use ``pd.NA`` as missing value indicator
        for the resulting DataFrame. (only applicable for the ``pyarrow``
        engine)
        As new dtypes are added that support ``pd.NA`` in the future, the
        output with this option will change to use those dtypes.
        Note: this is an experimental option, and behaviour (e.g. additional
        support dtypes) may change without notice.

        .. versionadded:: 1.2.0

    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    DataFrame
    """
    ...
