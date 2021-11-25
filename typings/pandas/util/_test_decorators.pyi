from contextlib import contextmanager
from typing import Callable

"""
This module provides decorator functions which can be applied to test objects
in order to skip those objects when certain conditions occur. A sample use case
is to detect if the platform is missing ``matplotlib``. If so, any test objects
which require ``matplotlib`` and decorated with ``@td.skip_if_no_mpl`` will be
skipped by ``pytest`` during the execution of the test suite.

To illustrate, after importing this module:

import pandas.util._test_decorators as td

The decorators can be applied to classes:

@td.skip_if_some_reason
class Foo:
    ...

Or individual functions:

@td.skip_if_some_reason
def test_foo():
    ...

For more information, refer to the ``pytest`` documentation on ``skipif``.
"""

def safe_import(mod_name: str, min_version: str | None = ...):  # -> Any | Literal[False]:
    """
    Parameters
    ----------
    mod_name : str
        Name of the module to be imported
    min_version : str, default None
        Minimum required version of the specified mod_name

    Returns
    -------
    object
        The imported module if successful, or False
    """
    ...

def skip_if_installed(package: str):  # -> MarkDecorator:
    """
    Skip a test if a package is installed.

    Parameters
    ----------
    package : str
        The name of the package.
    """
    ...

def skip_if_no(package: str, min_version: str | None = ...):  # -> MarkDecorator:
    """
    Generic function to help skip tests when required packages are not
    present on the testing system.

    This function returns a pytest mark with a skip condition that will be
    evaluated during test collection. An attempt will be made to import the
    specified ``package`` and optionally ensure it meets the ``min_version``

    The mark can be used as either a decorator for a test function or to be
    applied to parameters in pytest.mark.parametrize calls or parametrized
    fixtures.

    If the import and version check are unsuccessful, then the test function
    (or test case when used in conjunction with parametrization) will be
    skipped.

    Parameters
    ----------
    package: str
        The name of the required package.
    min_version: str or None, default None
        Optional minimum version of the package.

    Returns
    -------
    _pytest.mark.structures.MarkDecorator
        a pytest.mark.skipif to use as either a test decorator or a
        parametrization mark.
    """
    ...

skip_if_no_mpl = ...
skip_if_mpl = ...
skip_if_32bit = ...
skip_if_windows = ...
skip_if_windows_python_3 = ...
skip_if_has_locale = ...
skip_if_not_us_locale = ...
skip_if_no_scipy = ...
skip_if_no_ne = ...

def skip_if_np_lt(ver_str: str, *args, reason: str | None = ...): ...
def parametrize_fixture_doc(*args):  # -> (fixture: Unknown) -> Unknown:
    """
    Intended for use as a decorator for parametrized fixture,
    this function will wrap the decorated function with a pytest
    ``parametrize_fixture_doc`` mark. That mark will format
    initial fixture docstring by replacing placeholders {0}, {1} etc
    with parameters passed as arguments.

    Parameters
    ----------
    args: iterable
        Positional arguments for docstring.

    Returns
    -------
    function
        The decorated function wrapped within a pytest
        ``parametrize_fixture_doc`` mark
    """
    ...

def check_file_leaks(func) -> Callable:
    """
    Decorate a test function to check that we are not leaking file descriptors.
    """
    ...

@contextmanager
def file_leak_context():  # -> Generator[None, None, None]:
    """
    ContextManager analogue to check_file_leaks.
    """
    ...

def async_mark(): ...

skip_array_manager_not_yet_implemented = ...
skip_array_manager_invalid_test = ...
