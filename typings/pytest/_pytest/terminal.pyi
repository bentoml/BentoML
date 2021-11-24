
import argparse
import warnings
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

import attr
from _pytest._code import ExceptionInfo
from _pytest._code.code import ExceptionRepr
from _pytest.compat import final
from _pytest.config import Config, ExitCode, _PluggyPlugin, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.nodes import Item
from _pytest.reports import BaseReport, CollectReport, TestReport

"""Terminal reporting of the full testing process.

This is a good source for looking at the various reporting hooks.
"""
if TYPE_CHECKING:
    ...
REPORT_COLLECTING_RESOLUTION = ...
KNOWN_TYPES = ...
_REPORTCHARS_DEFAULT = ...
class MoreQuietAction(argparse.Action):
    """A modified copy of the argparse count action which counts down and updates
    the legacy quiet attribute at the same time.

    Used to unify verbosity handling.
    """
    def __init__(self, option_strings: Sequence[str], dest: str, default: object = ..., required: bool = ..., help: Optional[str] = ...) -> None:
        ...
    
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Union[str, Sequence[object], None], option_string: Optional[str] = ...) -> None:
        ...
    


def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_configure(config: Config) -> None:
    ...

def getreportopt(config: Config) -> str:
    ...

@hookimpl(trylast=True)
def pytest_report_teststatus(report: BaseReport) -> Tuple[str, str, str]:
    ...

@attr.s
class WarningReport:
    """Simple structure to hold warnings information captured by ``pytest_warning_recorded``.

    :ivar str message:
        User friendly message about the warning.
    :ivar str|None nodeid:
        nodeid that generated the warning (see ``get_location``).
    :ivar tuple|py.path.local fslocation:
        File system location of the source of the warning (see ``get_location``).
    """
    message = ...
    nodeid = ...
    fslocation = ...
    count_towards_summary = ...
    def get_location(self, config: Config) -> Optional[str]:
        """Return the more user-friendly information about the location of a warning, or None."""
        ...
    


@final
class TerminalReporter:
    def __init__(self, config: Config, file: Optional[TextIO] = ...) -> None:
        ...
    
    @property
    def verbosity(self) -> int:
        ...
    
    @property
    def showheader(self) -> bool:
        ...
    
    @property
    def no_header(self) -> bool:
        ...
    
    @property
    def no_summary(self) -> bool:
        ...
    
    @property
    def showfspath(self) -> bool:
        ...
    
    @showfspath.setter
    def showfspath(self, value: Optional[bool]) -> None:
        ...
    
    @property
    def showlongtestinfo(self) -> bool:
        ...
    
    def hasopt(self, char: str) -> bool:
        ...
    
    def write_fspath_result(self, nodeid: str, res, **markup: bool) -> None:
        ...
    
    def write_ensure_prefix(self, prefix: str, extra: str = ..., **kwargs) -> None:
        ...
    
    def ensure_newline(self) -> None:
        ...
    
    def write(self, content: str, *, flush: bool = ..., **markup: bool) -> None:
        ...
    
    def flush(self) -> None:
        ...
    
    def write_line(self, line: Union[str, bytes], **markup: bool) -> None:
        ...
    
    def rewrite(self, line: str, **markup: bool) -> None:
        """Rewinds the terminal cursor to the beginning and writes the given line.

        :param erase:
            If True, will also add spaces until the full terminal width to ensure
            previous lines are properly erased.

        The rest of the keyword arguments are markup instructions.
        """
        ...
    
    def write_sep(self, sep: str, title: Optional[str] = ..., fullwidth: Optional[int] = ..., **markup: bool) -> None:
        ...
    
    def section(self, title: str, sep: str = ..., **kw: bool) -> None:
        ...
    
    def line(self, msg: str, **kw: bool) -> None:
        ...
    
    def pytest_internalerror(self, excrepr: ExceptionRepr) -> bool:
        ...
    
    def pytest_warning_recorded(self, warning_message: warnings.WarningMessage, nodeid: str) -> None:
        ...
    
    def pytest_plugin_registered(self, plugin: _PluggyPlugin) -> None:
        ...
    
    def pytest_deselected(self, items: Sequence[Item]) -> None:
        ...
    
    def pytest_runtest_logstart(self, nodeid: str, location: Tuple[str, Optional[int], str]) -> None:
        ...
    
    def pytest_runtest_logreport(self, report: TestReport) -> None:
        ...
    
    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        ...
    
    def pytest_collection(self) -> None:
        ...
    
    def pytest_collectreport(self, report: CollectReport) -> None:
        ...
    
    def report_collect(self, final: bool = ...) -> None:
        ...
    
    @hookimpl(trylast=True)
    def pytest_sessionstart(self, session: Session) -> None:
        ...
    
    def pytest_report_header(self, config: Config) -> List[str]:
        ...
    
    def pytest_collection_finish(self, session: Session) -> None:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_sessionfinish(self, session: Session, exitstatus: Union[int, ExitCode]): # -> Generator[None, None, None]:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_terminal_summary(self) -> Generator[None, None, None]:
        ...
    
    def pytest_keyboard_interrupt(self, excinfo: ExceptionInfo[BaseException]) -> None:
        ...
    
    def pytest_unconfigure(self) -> None:
        ...
    
    def getreports(self, name: str): # -> list[Unknown]:
        ...
    
    def summary_warnings(self) -> None:
        ...
    
    def summary_passes(self) -> None:
        ...
    
    def print_teardown_sections(self, rep: TestReport) -> None:
        ...
    
    def summary_failures(self) -> None:
        ...
    
    def summary_errors(self) -> None:
        ...
    
    def summary_stats(self) -> None:
        ...
    
    def short_test_summary(self) -> None:
        ...
    
    def build_summary_stats_line(self) -> Tuple[List[Tuple[str, Dict[str, bool]]], str]:
        """
        Build the parts used in the last summary stats line.

        The summary stats line is the line shown at the end, "=== 12 passed, 2 errors in Xs===".

        This function builds a list of the "parts" that make up for the text in that line, in
        the example above it would be:

            [
                ("12 passed", {"green": True}),
                ("2 errors", {"red": True}
            ]

        That last dict for each line is a "markup dictionary", used by TerminalWriter to
        color output.

        The final color of the line is also determined by this function, and is the second
        element of the returned tuple.
        """
        ...
    


_color_for_type = ...
_color_for_type_default = ...
def pluralize(count: int, noun: str) -> Tuple[int, str]:
    ...

def format_session_duration(seconds: float) -> str:
    """Format the given seconds in a human readable manner to show in the final summary."""
    ...

