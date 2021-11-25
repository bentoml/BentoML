from contextlib import contextmanager
from typing import Any

@contextmanager
def decompress_file(path, compression):  # -> Generator[Buffer[Unknown], None, None]:
    """
    Open a compressed file and return a file object.

    Parameters
    ----------
    path : str
        The path where the file is read from.

    compression : {'gzip', 'bz2', 'zip', 'xz', None}
        Name of the decompression to use

    Returns
    -------
    file object
    """
    ...

@contextmanager
def set_timezone(tz: str):  # -> Generator[None, None, None]:
    """
    Context manager for temporarily setting a timezone.

    Parameters
    ----------
    tz : str
        A string representing a valid timezone.

    Examples
    --------
    >>> from datetime import datetime
    >>> from dateutil.tz import tzlocal
    >>> tzlocal().tzname(datetime.now())
    'IST'

    >>> with set_timezone('US/Eastern'):
    ...     tzlocal().tzname(datetime.now())
    ...
    'EDT'
    """
    ...

@contextmanager
def ensure_clean(
    filename=..., return_filelike: bool = ..., **kwargs: Any
):  # -> Generator[TextIOWrapper | str, None, None]:
    """
    Gets a temporary path and agrees to remove on close.

    This implementation does not use tempfile.mkstemp to avoid having a file handle.
    If the code using the returned path wants to delete the file itself, windows
    requires that no program has a file handle to it.

    Parameters
    ----------
    filename : str (optional)
        suffix of the created file.
    return_filelike : bool (default False)
        if True, returns a file-like which is *always* cleaned. Necessary for
        savefig and other functions which want to append extensions.
    **kwargs
        Additional keywords are passed to open().

    """
    ...

@contextmanager
def ensure_clean_dir():  # -> Generator[str, None, None]:
    """
    Get a temporary directory path and agrees to remove on close.

    Yields
    ------
    Temporary directory path
    """
    ...

@contextmanager
def ensure_safe_environment_variables():  # -> Generator[None, None, None]:
    """
    Get a context manager to safely set environment variables

    All changes will be undone on close, hence environment variables set
    within this contextmanager will neither persist nor change global state.
    """
    ...

@contextmanager
def with_csv_dialect(name, **kwargs):  # -> Generator[None, None, None]:
    """
    Context manager to temporarily register a CSV dialect for parsing CSV.

    Parameters
    ----------
    name : str
        The name of the dialect.
    kwargs : mapping
        The parameters for the dialect.

    Raises
    ------
    ValueError : the name of the dialect conflicts with a builtin one.

    See Also
    --------
    csv : Python's CSV library.
    """
    ...

@contextmanager
def use_numexpr(use, min_elements=...): ...

class RNGContext:
    """
    Context manager to set the numpy random number generator speed. Returns
    to the original value upon exiting the context manager.

    Parameters
    ----------
    seed : int
        Seed for numpy.random.seed

    Examples
    --------
    with RNGContext(42):
        np.random.randn()
    """

    def __init__(self, seed) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...
