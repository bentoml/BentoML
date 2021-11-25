from typing import Any

from pandas._typing import CompressionOptions, FilePathOrBuffer, StorageOptions
from pandas.core import generic
from pandas.util._decorators import doc

""" pickle compat """

@doc(storage_options=generic._shared_docs["storage_options"])
def to_pickle(
    obj: Any,
    filepath_or_buffer: FilePathOrBuffer,
    compression: CompressionOptions = ...,
    protocol: int = ...,
    storage_options: StorageOptions = ...,
):  # -> None:
    """
    Pickle (serialize) object to file.

    Parameters
    ----------
    obj : any object
        Any python object.
    filepath_or_buffer : str, path object or file-like object
        File path, URL, or buffer where the pickled object will be stored.

        .. versionchanged:: 1.0.0
           Accept URL. URL has to be of S3 or GCS.

    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default 'infer'
        If 'infer' and 'path_or_url' is path-like, then detect compression from
        the following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise no
        compression) If 'infer' and 'path_or_url' is not path-like, then use
        None (= no decompression).
    protocol : int
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL (see [1], paragraph 12.1.2). The possible
        values for this parameter depend on the version of Python. For Python
        2.x, possible values are 0, 1, 2. For Python>=3.0, 3 is a valid value.
        For Python >= 3.4, 4 is a valid value. A negative value for the
        protocol parameter is equivalent to setting its value to
        HIGHEST_PROTOCOL.

    {storage_options}

        .. versionadded:: 1.2.0

        .. [1] https://docs.python.org/3/library/pickle.html

    See Also
    --------
    read_pickle : Load pickled pandas object (or any object) from file.
    DataFrame.to_hdf : Write DataFrame to an HDF5 file.
    DataFrame.to_sql : Write DataFrame to a SQL database.
    DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

    Examples
    --------
    >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})
    >>> original_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")
    >>> unpickled_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9

    >>> import os
    >>> os.remove("./dummy.pkl")
    """
    ...

@doc(storage_options=generic._shared_docs["storage_options"])
def read_pickle(
    filepath_or_buffer: FilePathOrBuffer,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
):  # -> Any:
    """
    Load pickled pandas object (or any object) from file.

    .. warning::

       Loading pickled data received from untrusted sources can be
       unsafe. See `here <https://docs.python.org/3/library/pickle.html>`__.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        File path, URL, or buffer where the pickled object will be loaded from.

        .. versionchanged:: 1.0.0
           Accept URL. URL is not limited to S3 and GCS.

    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default 'infer'
        If 'infer' and 'path_or_url' is path-like, then detect compression from
        the following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise no
        compression) If 'infer' and 'path_or_url' is not path-like, then use
        None (= no decompression).

    {storage_options}

        .. versionadded:: 1.2.0

    Returns
    -------
    unpickled : same type as object stored in file

    See Also
    --------
    DataFrame.to_pickle : Pickle (serialize) DataFrame object to file.
    Series.to_pickle : Pickle (serialize) Series object to file.
    read_hdf : Read HDF5 file into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.
    read_parquet : Load a parquet object, returning a DataFrame.

    Notes
    -----
    read_pickle is only guaranteed to be backwards compatible to pandas 0.20.3.

    Examples
    --------
    >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})
    >>> original_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")
    >>> unpickled_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9

    >>> import os
    >>> os.remove("./dummy.pkl")
    """
    ...
