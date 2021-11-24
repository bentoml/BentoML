
from typing import Any, NoReturn, Optional

from typing_extensions import Protocol

"""Exception classes and constants handling test outcomes as well as
functions creating them."""
TYPE_CHECKING = ...
if TYPE_CHECKING:
    ...
else:
    ...
class OutcomeException(BaseException):
    """OutcomeException and its subclass instances indicate and contain info
    about test and collection outcomes."""
    def __init__(self, msg: Optional[str] = ..., pytrace: bool = ...) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    __str__ = ...


TEST_OUTCOME = ...
class Skipped(OutcomeException):
    __module__ = ...
    def __init__(self, msg: Optional[str] = ..., pytrace: bool = ..., allow_module_level: bool = ...) -> None:
        ...
    


class Failed(OutcomeException):
    """Raised from an explicit call to pytest.fail()."""
    __module__ = ...


class Exit(Exception):
    """Raised for immediate program exits (no tracebacks/summaries)."""
    def __init__(self, msg: str = ..., returncode: Optional[int] = ...) -> None:
        ...
    


_F = ...
_ET = ...
class _WithException(Protocol[_F, _ET]):
    Exception: _ET
    __call__: _F
    ...


@_with_exception(Exit)
def exit(msg: str, returncode: Optional[int] = ...) -> NoReturn:
    """Exit testing process.

    :param str msg: Message to display upon exit.
    :param int returncode: Return code to be used when exiting pytest.
    """
    ...

@_with_exception(Skipped)
def skip(msg: str = ..., *, allow_module_level: bool = ...) -> NoReturn:
    """Skip an executing test with the given message.

    This function should be called only during testing (setup, call or teardown) or
    during collection by using the ``allow_module_level`` flag.  This function can
    be called in doctests as well.

    :param bool allow_module_level:
        Allows this function to be called at module level, skipping the rest
        of the module. Defaults to False.

    .. note::
        It is better to use the :ref:`pytest.mark.skipif ref` marker when
        possible to declare a test to be skipped under certain conditions
        like mismatching platforms or dependencies.
        Similarly, use the ``# doctest: +SKIP`` directive (see `doctest.SKIP
        <https://docs.python.org/3/library/doctest.html#doctest.SKIP>`_)
        to skip a doctest statically.
    """
    ...

@_with_exception(Failed)
def fail(msg: str = ..., pytrace: bool = ...) -> NoReturn:
    """Explicitly fail an executing test with the given message.

    :param str msg:
        The message to show the user as reason for the failure.
    :param bool pytrace:
        If False, msg represents the full failure information and no
        python traceback will be reported.
    """
    ...

class XFailed(Failed):
    """Raised from an explicit call to pytest.xfail()."""
    ...


@_with_exception(XFailed)
def xfail(reason: str = ...) -> NoReturn:
    """Imperatively xfail an executing test or setup function with the given reason.

    This function should be called only during testing (setup, call or teardown).

    .. note::
        It is better to use the :ref:`pytest.mark.xfail ref` marker when
        possible to declare a test to be xfailed under certain conditions
        like known bugs or missing features.
    """
    ...

def importorskip(modname: str, minversion: Optional[str] = ..., reason: Optional[str] = ...) -> Any:
    """Import and return the requested module ``modname``, or skip the
    current test if the module cannot be imported.

    :param str modname:
        The name of the module to import.
    :param str minversion:
        If given, the imported module's ``__version__`` attribute must be at
        least this minimal version, otherwise the test is still skipped.
    :param str reason:
        If given, this reason is shown as the message when the module cannot
        be imported.

    :returns:
        The imported module. This should be assigned to its canonical name.

    Example::

        docutils = pytest.importorskip("docutils")
    """
    ...

