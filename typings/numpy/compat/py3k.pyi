import os

"""
Python 3.X compatibility tools.

While this file was originally intended for Python 2 -> 3 transition,
it is now used to create a compatibility layer between different
minor versions of Python 3.

While the active version of numpy may not support a given version of python, we
allow downstream libraries to continue to use these shims for forward
compatibility with numpy while they transition their code to newer versions of
Python.
"""
__all__ = [
    "bytes",
    "asbytes",
    "isfileobj",
    "getexception",
    "strchar",
    "unicode",
    "asunicode",
    "asbytes_nested",
    "asunicode_nested",
    "asstr",
    "open_latin1",
    "long",
    "basestring",
    "sixu",
    "integer_types",
    "is_pathlib_path",
    "npy_load_module",
    "Path",
    "pickle",
    "contextlib_nullcontext",
    "os_fspath",
    "os_PathLike",
]
long = int
integer_types = ...
basestring = str
unicode = str
bytes = bytes

def asunicode(s): ...
def asbytes(s): ...
def asstr(s): ...
def isfileobj(f): ...
def open_latin1(filename, mode=...): ...
def sixu(s): ...

strchar = ...

def getexception(): ...
def asbytes_nested(x): ...
def asunicode_nested(x): ...
def is_pathlib_path(obj):  # -> bool:
    """
    Check whether obj is a `pathlib.Path` object.

    Prefer using ``isinstance(obj, os.PathLike)`` instead of this function.
    """
    ...

class contextlib_nullcontext:
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True

    .. note::
        Prefer using `contextlib.nullcontext` instead of this context manager.
    """

    def __init__(self, enter_result=...) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *excinfo): ...

def npy_load_module(name, fn, info=...):  # -> ModuleType:
    """
    Load a module.

    .. versionadded:: 1.11.2

    Parameters
    ----------
    name : str
        Full module name.
    fn : str
        Path to module file.
    info : tuple, optional
        Only here for backward compatibility with Python 2.*.

    Returns
    -------
    mod : module

    """
    ...

os_fspath = ...
os_PathLike = os.PathLike
