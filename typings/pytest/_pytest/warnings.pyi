
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Optional

import pytest
from _pytest.config import Config
from _pytest.main import Session
from _pytest.nodes import Item
from _pytest.terminal import TerminalReporter
from typing_extensions import Literal

if TYPE_CHECKING:
    ...
def pytest_configure(config: Config) -> None:
    ...

@contextmanager
def catch_warnings_for_item(config: Config, ihook, when: Literal[config, collect, runtest], item: Optional[Item]) -> Generator[None, None, None]:
    """Context manager that catches warnings generated in the contained execution block.

    ``item`` can be None if we are not in the context of an item execution.

    Each warning captured triggers the ``pytest_warning_recorded`` hook.
    """
    ...

def warning_record_to_str(warning_message: warnings.WarningMessage) -> str:
    """Convert a warnings.WarningMessage to a string."""
    ...

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item: Item) -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection(session: Session) -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter: TerminalReporter) -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session: Session) -> Generator[None, None, None]:
    ...

@pytest.hookimpl(hookwrapper=True)
def pytest_load_initial_conftests(early_config: Config) -> Generator[None, None, None]:
    ...

