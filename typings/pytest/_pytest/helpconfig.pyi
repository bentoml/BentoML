
from argparse import Action
from typing import List, Optional, Union

import pytest
from _pytest.config import Config, ExitCode
from _pytest.config.argparsing import Parser

"""Version info, help messages, tracing configuration."""
class HelpAction(Action):
    """An argparse Action that will raise an exception in order to skip the
    rest of the argument parsing when --help is passed.

    This prevents argparse from quitting due to missing required arguments
    when any are defined, for example by ``pytest_addoption``.
    This is similar to the way that the builtin argparse --help option is
    implemented by raising SystemExit.
    """
    def __init__(self, option_strings, dest=..., default=..., help=...) -> None:
        ...
    
    def __call__(self, parser, namespace, values, option_string=...): # -> None:
        ...
    


def pytest_addoption(parser: Parser) -> None:
    ...

@pytest.hookimpl(hookwrapper=True)
def pytest_cmdline_parse(): # -> Generator[None, None, None]:
    ...

def showversion(config: Config) -> None:
    ...

def pytest_cmdline_main(config: Config) -> Optional[Union[int, ExitCode]]:
    ...

def showhelp(config: Config) -> None:
    ...

conftest_options = ...
def getpluginversioninfo(config: Config) -> List[str]:
    ...

def pytest_report_header(config: Config) -> List[str]:
    ...

