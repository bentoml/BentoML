
import logging
from contextlib import contextmanager
from io import StringIO
from typing import AbstractSet, Generator, List, Mapping, Optional, Tuple, Union

from _pytest import nodes
from _pytest._io import TerminalWriter
from _pytest.capture import CaptureManager
from _pytest.compat import final
from _pytest.config import Config, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest, fixture
from _pytest.main import Session
from _pytest.terminal import TerminalReporter

"""Access and control log capturing."""
DEFAULT_LOG_FORMAT = ...
DEFAULT_LOG_DATE_FORMAT = ...
_ANSI_ESCAPE_SEQ = ...
caplog_handler_key = ...
caplog_records_key = ...
class ColoredLevelFormatter(logging.Formatter):
    """A logging formatter which colorizes the %(levelname)..s part of the
    log format passed to __init__."""
    LOGLEVEL_COLOROPTS: Mapping[int, AbstractSet[str]] = ...
    LEVELNAME_FMT_REGEX = ...
    def __init__(self, terminalwriter: TerminalWriter, *args, **kwargs) -> None:
        ...
    
    def format(self, record: logging.LogRecord) -> str:
        ...
    


class PercentStyleMultiline(logging.PercentStyle):
    """A logging style with special support for multiline messages.

    If the message of a record consists of multiple lines, this style
    formats the message as if each line were logged separately.
    """
    def __init__(self, fmt: str, auto_indent: Union[int, str, bool, None]) -> None:
        ...
    
    def format(self, record: logging.LogRecord) -> str:
        ...
    


def get_option_ini(config: Config, *names: str): # -> Any | list[local] | list[str] | str | Notset | Literal[True] | None:
    ...

def pytest_addoption(parser: Parser) -> None:
    """Add options to control log capturing."""
    ...

_HandlerType = ...
class catching_logs:
    """Context manager that prepares the whole logging machinery properly."""
    __slots__ = ...
    def __init__(self, handler: _HandlerType, level: Optional[int] = ...) -> None:
        ...
    
    def __enter__(self): # -> _HandlerType@__init__:
        ...
    
    def __exit__(self, type, value, traceback): # -> None:
        ...
    


class LogCaptureHandler(logging.StreamHandler):
    """A logging handler that stores log records and the log text."""
    stream: StringIO
    def __init__(self) -> None:
        """Create a new log handler."""
        ...
    
    def emit(self, record: logging.LogRecord) -> None:
        """Keep the log records in a list in addition to the log text."""
        ...
    
    def reset(self) -> None:
        ...
    
    def handleError(self, record: logging.LogRecord) -> None:
        ...
    


@final
class LogCaptureFixture:
    """Provides access and control of log capturing."""
    def __init__(self, item: nodes.Node, *, _ispytest: bool = ...) -> None:
        ...
    
    @property
    def handler(self) -> LogCaptureHandler:
        """Get the logging handler used by the fixture.

        :rtype: LogCaptureHandler
        """
        ...
    
    def get_records(self, when: str) -> List[logging.LogRecord]:
        """Get the logging records for one of the possible test phases.

        :param str when:
            Which test phase to obtain the records from. Valid values are: "setup", "call" and "teardown".

        :returns: The list of captured records at the given stage.
        :rtype: List[logging.LogRecord]

        .. versionadded:: 3.4
        """
        ...
    
    @property
    def text(self) -> str:
        """The formatted log text."""
        ...
    
    @property
    def records(self) -> List[logging.LogRecord]:
        """The list of log records."""
        ...
    
    @property
    def record_tuples(self) -> List[Tuple[str, int, str]]:
        """A list of a stripped down version of log records intended
        for use in assertion comparison.

        The format of the tuple is:

            (logger_name, log_level, message)
        """
        ...
    
    @property
    def messages(self) -> List[str]:
        """A list of format-interpolated log messages.

        Unlike 'records', which contains the format string and parameters for
        interpolation, log messages in this list are all interpolated.

        Unlike 'text', which contains the output from the handler, log
        messages in this list are unadorned with levels, timestamps, etc,
        making exact comparisons more reliable.

        Note that traceback or stack info (from :func:`logging.exception` or
        the `exc_info` or `stack_info` arguments to the logging functions) is
        not included, as this is added by the formatter in the handler.

        .. versionadded:: 3.7
        """
        ...
    
    def clear(self) -> None:
        """Reset the list of log records and the captured log text."""
        ...
    
    def set_level(self, level: Union[int, str], logger: Optional[str] = ...) -> None:
        """Set the level of a logger for the duration of a test.

        .. versionchanged:: 3.4
            The levels of the loggers changed by this function will be
            restored to their initial values at the end of the test.

        :param int level: The level.
        :param str logger: The logger to update. If not given, the root logger.
        """
        ...
    
    @contextmanager
    def at_level(self, level: int, logger: Optional[str] = ...) -> Generator[None, None, None]:
        """Context manager that sets the level for capturing of logs. After
        the end of the 'with' statement the level is restored to its original
        value.

        :param int level: The level.
        :param str logger: The logger to update. If not given, the root logger.
        """
        ...
    


@fixture
def caplog(request: FixtureRequest) -> Generator[LogCaptureFixture, None, None]:
    """Access and control log capturing.

    Captured logs are available through the following properties/methods::

    * caplog.messages        -> list of format-interpolated log messages
    * caplog.text            -> string containing formatted log output
    * caplog.records         -> list of logging.LogRecord instances
    * caplog.record_tuples   -> list of (logger_name, level, message) tuples
    * caplog.clear()         -> clear captured records and formatted log output string
    """
    ...

def get_log_level_for_setting(config: Config, *setting_names: str) -> Optional[int]:
    ...

@hookimpl(trylast=True)
def pytest_configure(config: Config) -> None:
    ...

class LoggingPlugin:
    """Attaches to the logging module and captures log messages for each test."""
    def __init__(self, config: Config) -> None:
        """Create a new plugin to capture log messages.

        The formatter can be safely shared across all handlers so
        create a single one for the entire test session here.
        """
        ...
    
    def set_log_path(self, fname: str) -> None:
        """Set the filename parameter for Logging.FileHandler().

        Creates parent directory if it does not exist.

        .. warning::
            This is an experimental API.
        """
        ...
    
    @hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionstart(self) -> Generator[None, None, None]:
        ...
    
    @hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection(self) -> Generator[None, None, None]:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_runtestloop(self, session: Session) -> Generator[None, None, None]:
        ...
    
    @hookimpl
    def pytest_runtest_logstart(self) -> None:
        ...
    
    @hookimpl
    def pytest_runtest_logreport(self) -> None:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item: nodes.Item) -> Generator[None, None, None]:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: nodes.Item) -> Generator[None, None, None]:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item: nodes.Item) -> Generator[None, None, None]:
        ...
    
    @hookimpl
    def pytest_runtest_logfinish(self) -> None:
        ...
    
    @hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self) -> Generator[None, None, None]:
        ...
    
    @hookimpl
    def pytest_unconfigure(self) -> None:
        ...
    


class _FileHandler(logging.FileHandler):
    """A logging FileHandler with pytest tweaks."""
    def handleError(self, record: logging.LogRecord) -> None:
        ...
    


class _LiveLoggingStreamHandler(logging.StreamHandler):
    """A logging StreamHandler used by the live logging feature: it will
    write a newline before the first log message in each test.

    During live logging we must also explicitly disable stdout/stderr
    capturing otherwise it will get captured and won't appear in the
    terminal.
    """
    stream: TerminalReporter = ...
    def __init__(self, terminal_reporter: TerminalReporter, capture_manager: Optional[CaptureManager]) -> None:
        ...
    
    def reset(self) -> None:
        """Reset the handler; should be called before the start of each test."""
        ...
    
    def set_when(self, when: Optional[str]) -> None:
        """Prepare for the given test phase (setup/call/teardown)."""
        ...
    
    def emit(self, record: logging.LogRecord) -> None:
        ...
    
    def handleError(self, record: logging.LogRecord) -> None:
        ...
    


class _LiveLoggingNullHandler(logging.NullHandler):
    """A logging handler used when live logging is disabled."""
    def reset(self) -> None:
        ...
    
    def set_when(self, when: str) -> None:
        ...
    
    def handleError(self, record: logging.LogRecord) -> None:
        ...
    


