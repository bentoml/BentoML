from typing import TYPE_CHECKING

from pandas import DataFrame
from pandas._typing import FilePathOrBuffer

""" orc compat """
if TYPE_CHECKING: ...

def read_orc(
    path: FilePathOrBuffer, columns: list[str] | None = ..., **kwargs
) -> DataFrame:
    """
    Load an ORC object from the file path, returning a DataFrame.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.orc``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    columns : list, default None
        If not None, only these columns will be read from the file.
    **kwargs
        Any additional kwargs are passed to pyarrow.

    Returns
    -------
    DataFrame

    Notes
    -------
    Before using this function you should read the :ref:`user guide about ORC <io.orc>`
    and :ref:`install optional dependencies <install.warn_orc>`.
    """
    ...
