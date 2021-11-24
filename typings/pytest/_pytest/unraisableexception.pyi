
from types import TracebackType
from typing import Generator, Optional, Type

import pytest

class catch_unraisable_exception:
    """Context manager catching unraisable exception using sys.unraisablehook.

    Storing the exception value (cm.unraisable.exc_value) creates a reference
    cycle. The reference cycle is broken explicitly when the context manager
    exits.

    Storing the object (cm.unraisable.object) can resurrect it if it is set to
    an object which is being finalized. Exiting the context manager clears the
    stored object.

    Usage:
        with catch_unraisable_exception() as cm:
            # code creating an "unraisable exception"
            ...
            # check the unraisable exception: use cm.unraisable
            ...
        # cm.unraisable attribute no longer exists at this point
        # (to break a reference cycle)
    """
    def __init__(self) -> None:
        ...
    
    def __enter__(self) -> catch_unraisable_exception:
        ...
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        ...
    


def unraisable_exception_runtest_hook() -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup() -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call() -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown() -> Generator[None, None, None]:
    ...

