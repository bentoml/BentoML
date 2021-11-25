import platform
import sys
import warnings

from pandas._typing import F
from pandas.compat.numpy import (
    is_numpy_dev,
    np_array_datetime64_compat,
    np_datetime64_compat,
    np_version_under1p18,
    np_version_under1p19,
    np_version_under1p20,
)
from pandas.compat.pyarrow import (
    pa_version_under1p0,
    pa_version_under2p0,
    pa_version_under3p0,
    pa_version_under4p0,
)

"""
compat
======

Cross-compatible functions for different versions of Python.

Other items:
* platform checker
"""
PY38 = ...
PY39 = ...
PY310 = ...
PYPY = ...
IS64 = sys.maxsize > 2 ** 32

def set_function_name(f: F, name: str, cls) -> F:
    """
    Bind the name/qualname attributes of the function.
    """
    ...

def is_platform_little_endian() -> bool:
    """
    Checking if the running platform is little endian.

    Returns
    -------
    bool
        True if the running platform is little endian.
    """
    ...

def is_platform_windows() -> bool:
    """
    Checking if the running platform is windows.

    Returns
    -------
    bool
        True if the running platform is windows.
    """
    ...

def is_platform_linux() -> bool:
    """
    Checking if the running platform is linux.

    Returns
    -------
    bool
        True if the running platform is linux.
    """
    ...

def is_platform_mac() -> bool:
    """
    Checking if the running platform is mac.

    Returns
    -------
    bool
        True if the running platform is mac.
    """
    ...

def is_platform_arm() -> bool:
    """
    Checking if he running platform use ARM architecture.

    Returns
    -------
    bool
        True if the running platform uses ARM architecture.
    """
    ...

def import_lzma():  # -> Module("lzma") | None:
    """
    Importing the `lzma` module.

    Warns
    -----
    When the `lzma` module is not available.
    """
    ...

def get_lzma_file(lzma):
    """
    Importing the `LZMAFile` class from the `lzma` module.

    Returns
    -------
    class
        The `LZMAFile` class from the `lzma` module.

    Raises
    ------
    RuntimeError
        If the `lzma` module was not imported correctly, or didn't exist.
    """
    ...

__all__ = [
    "is_numpy_dev",
    "np_array_datetime64_compat",
    "np_datetime64_compat",
    "np_version_under1p18",
    "np_version_under1p19",
    "np_version_under1p20",
    "pa_version_under1p0",
    "pa_version_under2p0",
    "pa_version_under3p0",
    "pa_version_under4p0",
]
