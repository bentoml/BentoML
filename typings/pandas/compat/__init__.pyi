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

PY38 = ...
PY39 = ...
PY310 = ...
PYPY = ...
IS64 = sys.maxsize > 2 ** 32

def set_function_name(f: F, name: str, cls) -> F: ...
def is_platform_little_endian() -> bool: ...
def is_platform_windows() -> bool: ...
def is_platform_linux() -> bool: ...
def is_platform_mac() -> bool: ...
def is_platform_arm() -> bool: ...
def import_lzma(): ...
def get_lzma_file(lzma): ...

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
