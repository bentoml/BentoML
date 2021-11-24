
import argparse
import collections.abc
import contextlib
import copy
import enum
import inspect
import os
import re
import shlex
import sys
import types
import warnings
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
    Union,
)

import _pytest._code
import _pytest.deprecated
import _pytest.hookspec
import attr
import py
from _pytest._code import ExceptionInfo, filter_traceback
from _pytest._code.code import _TracebackStyle
from _pytest._io import TerminalWriter
from _pytest.compat import final, importlib_metadata
from _pytest.outcomes import Skipped, fail
from _pytest.pathlib import ImportMode, bestrelpath, import_path
from _pytest.store import Store
from _pytest.terminal import TerminalReporter
from _pytest.warning_types import PytestConfigWarning
from pluggy import HookimplMarker, HookspecMarker, PluginManager

from .argparsing import Argument
from .exceptions import PrintHelp as PrintHelp
from .exceptions import UsageError as UsageError
from .findpaths import determine_setup

"""Command line options, ini-file and conftest.py processing."""
if TYPE_CHECKING:
    ...
_PluggyPlugin = object
hookimpl = ...
hookspec = ...
@final
class ExitCode(enum.IntEnum):
    """Encodes the valid exit codes by pytest.

    Currently users and plugins may supply other exit codes as well.

    .. versionadded:: 5.0
    """
    OK = ...
    TESTS_FAILED = ...
    INTERRUPTED = ...
    INTERNAL_ERROR = ...
    USAGE_ERROR = ...
    NO_TESTS_COLLECTED = ...


class ConftestImportFailure(Exception):
    def __init__(self, path: py.path.local, excinfo: Tuple[Type[Exception], Exception, TracebackType]) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


def filter_traceback_for_conftest_import_failure(entry: _pytest._code.TracebackEntry) -> bool:
    """Filter tracebacks entries which point to pytest internals or importlib.

    Make a special case for importlib because we use it to import test modules and conftest files
    in _pytest.pathlib.import_path.
    """
    ...

def main(args: Optional[Union[List[str], py.path.local]] = ..., plugins: Optional[Sequence[Union[str, _PluggyPlugin]]] = ...) -> Union[int, ExitCode]:
    """Perform an in-process test run.

    :param args: List of command line arguments.
    :param plugins: List of plugin objects to be auto-registered during initialization.

    :returns: An exit code.
    """
    ...

def console_main() -> int:
    """The CLI entry point of pytest.

    This function is not meant for programmable use; use `main()` instead.
    """
    ...

class cmdline:
    main = ...


def filename_arg(path: str, optname: str) -> str:
    """Argparse type validator for filename arguments.

    :path: Path of filename.
    :optname: Name of the option.
    """
    ...

def directory_arg(path: str, optname: str) -> str:
    """Argparse type validator for directory arguments.

    :path: Path of directory.
    :optname: Name of the option.
    """
    ...

essential_plugins = ...
default_plugins = ...
builtin_plugins = ...
def get_config(args: Optional[List[str]] = ..., plugins: Optional[Sequence[Union[str, _PluggyPlugin]]] = ...) -> Config:
    ...

def get_plugin_manager() -> PytestPluginManager:
    """Obtain a new instance of the
    :py:class:`_pytest.config.PytestPluginManager`, with default plugins
    already loaded.

    This function can be used by integration with other tools, like hooking
    into pytest to run tests into an IDE.
    """
    ...

@final
class PytestPluginManager(PluginManager):
    """A :py:class:`pluggy.PluginManager <pluggy.PluginManager>` with
    additional pytest-specific functionality:

    * Loading plugins from the command line, ``PYTEST_PLUGINS`` env variable and
      ``pytest_plugins`` global variables found in plugins being loaded.
    * ``conftest.py`` loading during start-up.
    """
    def __init__(self) -> None:
        ...
    
    def parse_hookimpl_opts(self, plugin: _PluggyPlugin, name: str): # -> dict[Unknown, Unknown] | None:
        ...
    
    def parse_hookspec_opts(self, module_or_class, name: str): # -> dict[str, bool]:
        ...
    
    def register(self, plugin: _PluggyPlugin, name: Optional[str] = ...) -> Optional[str]:
        ...
    
    def getplugin(self, name: str): # -> _PluggyPlugin | None:
        ...
    
    def hasplugin(self, name: str) -> bool:
        """Return whether a plugin with the given name is registered."""
        ...
    
    def pytest_configure(self, config: Config) -> None:
        """:meta private:"""
        ...
    
    def consider_preparse(self, args: Sequence[str], *, exclude_only: bool = ...) -> None:
        ...
    
    def consider_pluginarg(self, arg: str) -> None:
        ...
    
    def consider_conftest(self, conftestmodule: types.ModuleType) -> None:
        ...
    
    def consider_env(self) -> None:
        ...
    
    def consider_module(self, mod: types.ModuleType) -> None:
        ...
    
    def import_plugin(self, modname: str, consider_entry_points: bool = ...) -> None:
        """Import a plugin with ``modname``.

        If ``consider_entry_points`` is True, entry point names are also
        considered to find a plugin.
        """
        ...
    


class Notset:
    def __repr__(self): # -> Literal['<NOTSET>']:
        ...
    


notset = ...
@final
class Config:
    """Access to configuration values, pluginmanager and plugin hooks.

    :param PytestPluginManager pluginmanager:

    :param InvocationParams invocation_params:
        Object containing parameters regarding the :func:`pytest.main`
        invocation.
    """
    @final
    @attr.s(frozen=True)
    class InvocationParams:
        """Holds parameters passed during :func:`pytest.main`.

        The object attributes are read-only.

        .. versionadded:: 5.1

        .. note::

            Note that the environment variable ``PYTEST_ADDOPTS`` and the ``addopts``
            ini option are handled by pytest, not being included in the ``args`` attribute.

            Plugins accessing ``InvocationParams`` must be aware of that.
        """
        args = ...
        plugins = ...
        dir = ...
    
    
    def __init__(self, pluginmanager: PytestPluginManager, *, invocation_params: Optional[InvocationParams] = ...) -> None:
        ...
    
    @property
    def invocation_dir(self) -> py.path.local:
        """The directory from which pytest was invoked.

        Prefer to use :attr:`invocation_params.dir <InvocationParams.dir>`,
        which is a :class:`pathlib.Path`.

        :type: py.path.local
        """
        ...
    
    @property
    def rootpath(self) -> Path:
        """The path to the :ref:`rootdir <rootdir>`.

        :type: pathlib.Path

        .. versionadded:: 6.1
        """
        ...
    
    @property
    def rootdir(self) -> py.path.local:
        """The path to the :ref:`rootdir <rootdir>`.

        Prefer to use :attr:`rootpath`, which is a :class:`pathlib.Path`.

        :type: py.path.local
        """
        ...
    
    @property
    def inipath(self) -> Optional[Path]:
        """The path to the :ref:`configfile <configfiles>`.

        :type: Optional[pathlib.Path]

        .. versionadded:: 6.1
        """
        ...
    
    @property
    def inifile(self) -> Optional[py.path.local]:
        """The path to the :ref:`configfile <configfiles>`.

        Prefer to use :attr:`inipath`, which is a :class:`pathlib.Path`.

        :type: Optional[py.path.local]
        """
        ...
    
    def add_cleanup(self, func: Callable[[], None]) -> None:
        """Add a function to be called when the config object gets out of
        use (usually coninciding with pytest_unconfigure)."""
        ...
    
    def get_terminal_writer(self) -> TerminalWriter:
        ...
    
    def pytest_cmdline_parse(self, pluginmanager: PytestPluginManager, args: List[str]) -> Config:
        ...
    
    def notify_exception(self, excinfo: ExceptionInfo[BaseException], option: Optional[argparse.Namespace] = ...) -> None:
        ...
    
    def cwd_relative_nodeid(self, nodeid: str) -> str:
        ...
    
    @classmethod
    def fromdictargs(cls, option_dict, args) -> Config:
        """Constructor usable for subprocesses."""
        ...
    
    @hookimpl(trylast=True)
    def pytest_load_initial_conftests(self, early_config: Config) -> None:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_collection(self) -> Generator[None, None, None]:
        """Validate invalid ini keys after collection is done so we take in account
        options added by late-loading conftest files."""
        ...
    
    def parse(self, args: List[str], addopts: bool = ...) -> None:
        ...
    
    def issue_config_time_warning(self, warning: Warning, stacklevel: int) -> None:
        """Issue and handle a warning during the "configure" stage.

        During ``pytest_configure`` we can't capture warnings using the ``catch_warnings_for_item``
        function because it is not possible to have hookwrappers around ``pytest_configure``.

        This function is mainly intended for plugins that need to issue warnings during
        ``pytest_configure`` (or similar stages).

        :param warning: The warning instance.
        :param stacklevel: stacklevel forwarded to warnings.warn.
        """
        ...
    
    def addinivalue_line(self, name: str, line: str) -> None:
        """Add a line to an ini-file option. The option must have been
        declared but might not yet be set in which case the line becomes
        the first line in its value."""
        ...
    
    def getini(self, name: str): # -> Any | list[local] | list[str] | bool | str:
        """Return configuration value from an :ref:`ini file <configfiles>`.

        If the specified name hasn't been registered through a prior
        :py:func:`parser.addini <_pytest.config.argparsing.Parser.addini>`
        call (usually from a plugin), a ValueError is raised.
        """
        ...
    
    def getoption(self, name: str, default=..., skip: bool = ...): # -> Any:
        """Return command line option value.

        :param name: Name of the option.  You may also specify
            the literal ``--OPT`` option instead of the "dest" option name.
        :param default: Default value if no option of that name exists.
        :param skip: If True, raise pytest.skip if option does not exists
            or has a None value.
        """
        ...
    
    def getvalue(self, name: str, path=...): # -> Any | Notset:
        """Deprecated, use getoption() instead."""
        ...
    
    def getvalueorskip(self, name: str, path=...): # -> Any | Notset:
        """Deprecated, use getoption(skip=True) instead."""
        ...
    


def create_terminal_writer(config: Config, file: Optional[TextIO] = ...) -> TerminalWriter:
    """Create a TerminalWriter instance configured according to the options
    in the config object.

    Every code which requires a TerminalWriter object and has access to a
    config object should use this function.
    """
    ...

@lru_cache(maxsize=50)
def parse_warning_filter(arg: str, *, escape: bool) -> Tuple[str, str, Type[Warning], str, int]:
    """Parse a warnings filter string.

    This is copied from warnings._setoption, but does not apply the filter,
    only parses it, and makes the escaping optional.
    """
    ...

def apply_warning_filters(config_filters: Iterable[str], cmdline_filters: Iterable[str]) -> None:
    """Applies pytest-configured filters to the warnings module"""
    ...

