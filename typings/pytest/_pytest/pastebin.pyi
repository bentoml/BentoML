
from typing import Union

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.terminal import TerminalReporter

"""Submit failure or test session information to a pastebin service."""
pastebinfile_key = ...
def pytest_addoption(parser: Parser) -> None:
    ...

@pytest.hookimpl(trylast=True)
def pytest_configure(config: Config) -> None:
    ...

def pytest_unconfigure(config: Config) -> None:
    ...

def create_new_paste(contents: Union[str, bytes]) -> str:
    """Create a new paste using the bpaste.net service.

    :contents: Paste contents string.
    :returns: URL to the pasted contents, or an error message.
    """
    ...

def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    ...

