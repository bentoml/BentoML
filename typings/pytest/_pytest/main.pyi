
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import attr
import py
from _pytest import nodes
from _pytest.compat import final
from _pytest.config import Config, ExitCode, PytestPluginManager, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureManager
from _pytest.reports import CollectReport, TestReport
from _pytest.runner import SetupState
from typing_extensions import Literal

"""Core implementation of the testing process: init, session, runtest loop."""
if TYPE_CHECKING:
    ...
def pytest_addoption(parser: Parser) -> None:
    ...

def validate_basetemp(path: str) -> str:
    ...

def wrap_session(config: Config, doit: Callable[[Config, Session], Optional[Union[int, ExitCode]]]) -> Union[int, ExitCode]:
    """Skeleton command line program."""
    ...

def pytest_cmdline_main(config: Config) -> Union[int, ExitCode]:
    ...

def pytest_collection(session: Session) -> None:
    ...

def pytest_runtestloop(session: Session) -> bool:
    ...

def pytest_ignore_collect(path: py.path.local, config: Config) -> Optional[bool]:
    ...

def pytest_collection_modifyitems(items: List[nodes.Item], config: Config) -> None:
    ...

class FSHookProxy:
    def __init__(self, pm: PytestPluginManager, remove_mods) -> None:
        ...
    
    def __getattr__(self, name: str):
        ...
    


class Interrupted(KeyboardInterrupt):
    """Signals that the test run was interrupted."""
    __module__ = ...


class Failed(Exception):
    """Signals a stop as failed test run."""
    ...


@attr.s
class _bestrelpath_cache(Dict[Path, str]):
    path = ...
    def __missing__(self, path: Path) -> str:
        ...
    


@final
class Session(nodes.FSCollector):
    Interrupted = Interrupted
    Failed = Failed
    _setupstate: SetupState
    _fixturemanager: FixtureManager
    exitstatus: Union[int, ExitCode]
    def __init__(self, config: Config) -> None:
        ...
    
    @classmethod
    def from_config(cls, config: Config) -> Session:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @hookimpl(tryfirst=True)
    def pytest_collectstart(self) -> None:
        ...
    
    @hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: Union[TestReport, CollectReport]) -> None:
        ...
    
    pytest_collectreport = ...
    def isinitpath(self, path: py.path.local) -> bool:
        ...
    
    def gethookproxy(self, fspath: py.path.local): # -> FSHookProxy:
        ...
    
    @overload
    def perform_collect(self, args: Optional[Sequence[str]] = ..., genitems: Literal[True] = ...) -> Sequence[nodes.Item]:
        ...
    
    @overload
    def perform_collect(self, args: Optional[Sequence[str]] = ..., genitems: bool = ...) -> Sequence[Union[nodes.Item, nodes.Collector]]:
        ...
    
    def perform_collect(self, args: Optional[Sequence[str]] = ..., genitems: bool = ...) -> Sequence[Union[nodes.Item, nodes.Collector]]:
        """Perform the collection phase for this session.

        This is called by the default
        :func:`pytest_collection <_pytest.hookspec.pytest_collection>` hook
        implementation; see the documentation of this hook for more details.
        For testing purposes, it may also be called directly on a fresh
        ``Session``.

        This function normally recursively expands any collectors collected
        from the session to their items, and only items are returned. For
        testing purposes, this may be suppressed by passing ``genitems=False``,
        in which case the return value contains these collectors unexpanded,
        and ``session.items`` is empty.
        """
        ...
    
    def collect(self) -> Iterator[Union[nodes.Item, nodes.Collector]]:
        ...
    
    def genitems(self, node: Union[nodes.Item, nodes.Collector]) -> Iterator[nodes.Item]:
        ...
    


def search_pypath(module_name: str) -> str:
    """Search sys.path for the given a dotted module name, and return its file system path."""
    ...

def resolve_collection_argument(invocation_path: Path, arg: str, *, as_pypath: bool = ...) -> Tuple[py.path.local, List[str]]:
    """Parse path arguments optionally containing selection parts and return (fspath, names).

    Command-line arguments can point to files and/or directories, and optionally contain
    parts for specific tests selection, for example:

        "pkg/tests/test_foo.py::TestClass::test_foo"

    This function ensures the path exists, and returns a tuple:

        (py.path.path("/full/path/to/pkg/tests/test_foo.py"), ["TestClass", "test_foo"])

    When as_pypath is True, expects that the command-line argument actually contains
    module paths instead of file-system paths:

        "pkg.tests.test_foo::TestClass::test_foo"

    In which case we search sys.path for a matching module, and then return the *path* to the
    found module.

    If the path doesn't exist, raise UsageError.
    If the path is a directory and selection parts are present, raise UsageError.
    """
    ...

