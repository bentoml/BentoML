
from types import TracebackType
from typing import Generator, Optional, Type

import pytest

class catch_threading_exception:
    """Context manager catching threading.Thread exception using
    threading.excepthook.

    Storing exc_value using a custom hook can create a reference cycle. The
    reference cycle is broken explicitly when the context manager exits.

    Storing thread using a custom hook can resurrect it if it is set to an
    object which is being finalized. Exiting the context manager clears the
    stored object.

    Usage:
        with threading_helper.catch_threading_exception() as cm:
            # code spawning a thread which raises an exception
            ...
            # check the thread exception: use cm.args
            ...
        # cm.args attribute no longer exists at this point
        # (to break a reference cycle)
    """
    def __init__(self) -> None:
        ...
    
    def __enter__(self) -> catch_threading_exception:
        ...
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        ...
    


def thread_exception_runtest_hook() -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_setup() -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call() -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown() -> Generator[None, None, None]:
    ...

