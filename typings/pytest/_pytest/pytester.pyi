
import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Type,
    Union,
    overload,
)

import attr
import pexpect
import py
from _pytest.compat import final
from _pytest.config import Config, ExitCode, PytestPluginManager, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest, fixture
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector, Item
from _pytest.reports import CollectReport, TestReport
from _pytest.tmpdir import TempPathFactory
from iniconfig import SectionWrapper
from typing_extensions import Literal

"""(Disabled by default) support for testing pytest and pytest plugins.

PYTEST_DONT_REWRITE
"""
if TYPE_CHECKING:
    ...
pytest_plugins = ...
IGNORE_PAM = ...
def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_configure(config: Config) -> None:
    ...

class LsofFdLeakChecker:
    def get_open_files(self) -> List[Tuple[str, str]]:
        ...
    
    def matching_platform(self) -> bool:
        ...
    
    @hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item: Item) -> Generator[None, None, None]:
        ...
    


class PytestArg:
    def __init__(self, request: FixtureRequest) -> None:
        ...
    
    def gethookrecorder(self, hook) -> HookRecorder:
        ...
    


def get_public_names(values: Iterable[str]) -> List[str]:
    """Only return names from iterator values without a leading underscore."""
    ...

class ParsedCall:
    def __init__(self, name: str, kwargs) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    if TYPE_CHECKING:
        def __getattr__(self, key: str): # -> None:
            ...
        


class HookRecorder:
    """Record all hooks called in a plugin manager.

    This wraps all the hook calls in the plugin manager, recording each call
    before propagating the normal calls.
    """
    def __init__(self, pluginmanager: PytestPluginManager) -> None:
        ...
    
    def finish_recording(self) -> None:
        ...
    
    def getcalls(self, names: Union[str, Iterable[str]]) -> List[ParsedCall]:
        ...
    
    def assert_contains(self, entries: Sequence[Tuple[str, str]]) -> None:
        ...
    
    def popcall(self, name: str) -> ParsedCall:
        ...
    
    def getcall(self, name: str) -> ParsedCall:
        ...
    
    @overload
    def getreports(self, names: Literal[pytest_collectreport]) -> Sequence[CollectReport]:
        ...
    
    @overload
    def getreports(self, names: Literal[pytest_runtest_logreport]) -> Sequence[TestReport]:
        ...
    
    @overload
    def getreports(self, names: Union[str, Iterable[str]] = ...) -> Sequence[Union[CollectReport, TestReport]]:
        ...
    
    def getreports(self, names: Union[str, Iterable[str]] = ...) -> Sequence[Union[CollectReport, TestReport]]:
        ...
    
    def matchreport(self, inamepart: str = ..., names: Union[str, Iterable[str]] = ..., when: Optional[str] = ...) -> Union[CollectReport, TestReport]:
        """Return a testreport whose dotted import path matches."""
        ...
    
    @overload
    def getfailures(self, names: Literal[pytest_collectreport]) -> Sequence[CollectReport]:
        ...
    
    @overload
    def getfailures(self, names: Literal[pytest_runtest_logreport]) -> Sequence[TestReport]:
        ...
    
    @overload
    def getfailures(self, names: Union[str, Iterable[str]] = ...) -> Sequence[Union[CollectReport, TestReport]]:
        ...
    
    def getfailures(self, names: Union[str, Iterable[str]] = ...) -> Sequence[Union[CollectReport, TestReport]]:
        ...
    
    def getfailedcollections(self) -> Sequence[CollectReport]:
        ...
    
    def listoutcomes(self) -> Tuple[Sequence[TestReport], Sequence[Union[CollectReport, TestReport]], Sequence[Union[CollectReport, TestReport]]],:
        ...
    
    def countoutcomes(self) -> List[int]:
        ...
    
    def assertoutcome(self, passed: int = ..., skipped: int = ..., failed: int = ...) -> None:
        ...
    
    def clear(self) -> None:
        ...
    


@fixture
def linecomp() -> LineComp:
    """A :class: `LineComp` instance for checking that an input linearly
    contains a sequence of strings."""
    ...

@fixture(name="LineMatcher")
def LineMatcher_fixture(request: FixtureRequest) -> Type[LineMatcher]:
    """A reference to the :class: `LineMatcher`.

    This is instantiable with a list of lines (without their trailing newlines).
    This is useful for testing large texts, such as the output of commands.
    """
    ...

@fixture
def pytester(request: FixtureRequest, tmp_path_factory: TempPathFactory) -> Pytester:
    """
    Facilities to write tests/configuration files, execute pytest in isolation, and match
    against expected output, perfect for black-box testing of pytest plugins.

    It attempts to isolate the test run from external factors as much as possible, modifying
    the current working directory to ``path`` and environment variables during initialization.

    It is particularly useful for testing plugins. It is similar to the :fixture:`tmp_path`
    fixture but provides methods which aid in testing pytest itself.
    """
    ...

@fixture
def testdir(pytester: Pytester) -> Testdir:
    """
    Identical to :fixture:`pytester`, and provides an instance whose methods return
    legacy ``py.path.local`` objects instead when applicable.

    New code should avoid using :fixture:`testdir` in favor of :fixture:`pytester`.
    """
    ...

rex_session_duration = ...
rex_outcome = ...
class RunResult:
    """The result of running a command."""
    def __init__(self, ret: Union[int, ExitCode], outlines: List[str], errlines: List[str], duration: float) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def parseoutcomes(self) -> Dict[str, int]:
        """Return a dictionary of outcome noun -> count from parsing the terminal
        output that the test process produced.

        The returned nouns will always be in plural form::

            ======= 1 failed, 1 passed, 1 warning, 1 error in 0.13s ====

        Will return ``{"failed": 1, "passed": 1, "warnings": 1, "errors": 1}``.
        """
        ...
    
    @classmethod
    def parse_summary_nouns(cls, lines) -> Dict[str, int]:
        """Extract the nouns from a pytest terminal summary line.

        It always returns the plural noun for consistency::

            ======= 1 failed, 1 passed, 1 warning, 1 error in 0.13s ====

        Will return ``{"failed": 1, "passed": 1, "warnings": 1, "errors": 1}``.
        """
        ...
    
    def assert_outcomes(self, passed: int = ..., skipped: int = ..., failed: int = ..., errors: int = ..., xpassed: int = ..., xfailed: int = ...) -> None:
        """Assert that the specified outcomes appear with the respective
        numbers (0 means it didn't occur) in the text output from a test run."""
        ...
    


class CwdSnapshot:
    def __init__(self) -> None:
        ...
    
    def restore(self) -> None:
        ...
    


class SysModulesSnapshot:
    def __init__(self, preserve: Optional[Callable[[str], bool]] = ...) -> None:
        ...
    
    def restore(self) -> None:
        ...
    


class SysPathsSnapshot:
    def __init__(self) -> None:
        ...
    
    def restore(self) -> None:
        ...
    


@final
class Pytester:
    """
    Facilities to write tests/configuration files, execute pytest in isolation, and match
    against expected output, perfect for black-box testing of pytest plugins.

    It attempts to isolate the test run from external factors as much as possible, modifying
    the current working directory to ``path`` and environment variables during initialization.

    Attributes:

    :ivar Path path: temporary directory path used to create files/run tests from, etc.

    :ivar plugins:
       A list of plugins to use with :py:meth:`parseconfig` and
       :py:meth:`runpytest`.  Initially this is an empty list but plugins can
       be added to the list.  The type of items to add to the list depends on
       the method using them so refer to them for details.
    """
    __test__ = ...
    CLOSE_STDIN = object
    class TimeoutExpired(Exception):
        ...
    
    
    def __init__(self, request: FixtureRequest, tmp_path_factory: TempPathFactory, *, _ispytest: bool = ...) -> None:
        ...
    
    @property
    def path(self) -> Path:
        """Temporary directory where files are created and pytest is executed."""
        ...
    
    def __repr__(self) -> str:
        ...
    
    def make_hook_recorder(self, pluginmanager: PytestPluginManager) -> HookRecorder:
        """Create a new :py:class:`HookRecorder` for a PluginManager."""
        ...
    
    def chdir(self) -> None:
        """Cd into the temporary directory.

        This is done automatically upon instantiation.
        """
        ...
    
    def makefile(self, ext: str, *args: str, **kwargs: str) -> Path:
        r"""Create new file(s) in the test directory.

        :param str ext:
            The extension the file(s) should use, including the dot, e.g. `.py`.
        :param args:
            All args are treated as strings and joined using newlines.
            The result is written as contents to the file.  The name of the
            file is based on the test function requesting this fixture.
        :param kwargs:
            Each keyword is the name of a file, while the value of it will
            be written as contents of the file.

        Examples:

        .. code-block:: python

            pytester.makefile(".txt", "line1", "line2")

            pytester.makefile(".ini", pytest="[pytest]\naddopts=-rs\n")

        """
        ...
    
    def makeconftest(self, source: str) -> Path:
        """Write a contest.py file with 'source' as contents."""
        ...
    
    def makeini(self, source: str) -> Path:
        """Write a tox.ini file with 'source' as contents."""
        ...
    
    def getinicfg(self, source: str) -> SectionWrapper:
        """Return the pytest section from the tox.ini config file."""
        ...
    
    def makepyprojecttoml(self, source: str) -> Path:
        """Write a pyproject.toml file with 'source' as contents.

        .. versionadded:: 6.0
        """
        ...
    
    def makepyfile(self, *args, **kwargs) -> Path:
        r"""Shortcut for .makefile() with a .py extension.

        Defaults to the test name with a '.py' extension, e.g test_foobar.py, overwriting
        existing files.

        Examples:

        .. code-block:: python

            def test_something(pytester):
                # Initial file is created test_something.py.
                pytester.makepyfile("foobar")
                # To create multiple files, pass kwargs accordingly.
                pytester.makepyfile(custom="foobar")
                # At this point, both 'test_something.py' & 'custom.py' exist in the test directory.

        """
        ...
    
    def maketxtfile(self, *args, **kwargs) -> Path:
        r"""Shortcut for .makefile() with a .txt extension.

        Defaults to the test name with a '.txt' extension, e.g test_foobar.txt, overwriting
        existing files.

        Examples:

        .. code-block:: python

            def test_something(pytester):
                # Initial file is created test_something.txt.
                pytester.maketxtfile("foobar")
                # To create multiple files, pass kwargs accordingly.
                pytester.maketxtfile(custom="foobar")
                # At this point, both 'test_something.txt' & 'custom.txt' exist in the test directory.

        """
        ...
    
    def syspathinsert(self, path: Optional[Union[str, os.PathLike[str]]] = ...) -> None:
        """Prepend a directory to sys.path, defaults to :py:attr:`tmpdir`.

        This is undone automatically when this object dies at the end of each
        test.
        """
        ...
    
    def mkdir(self, name: str) -> Path:
        """Create a new (sub)directory."""
        ...
    
    def mkpydir(self, name: str) -> Path:
        """Create a new python package.

        This creates a (sub)directory with an empty ``__init__.py`` file so it
        gets recognised as a Python package.
        """
        ...
    
    def copy_example(self, name: Optional[str] = ...) -> Path:
        """Copy file from project's directory into the testdir.

        :param str name: The name of the file to copy.
        :return: path to the copied directory (inside ``self.path``).

        """
        ...
    
    Session = Session
    def getnode(self, config: Config, arg: Union[str, os.PathLike[str]]) -> Optional[Union[Collector, Item]]:
        """Return the collection node of a file.

        :param _pytest.config.Config config:
           A pytest config.
           See :py:meth:`parseconfig` and :py:meth:`parseconfigure` for creating it.
        :param py.path.local arg:
            Path to the file.
        """
        ...
    
    def getpathnode(self, path: Union[str, os.PathLike[str]]): # -> Item | Collector:
        """Return the collection node of a file.

        This is like :py:meth:`getnode` but uses :py:meth:`parseconfigure` to
        create the (configured) pytest Config instance.

        :param py.path.local path: Path to the file.
        """
        ...
    
    def genitems(self, colitems: Sequence[Union[Item, Collector]]) -> List[Item]:
        """Generate all test items from a collection node.

        This recurses into the collection node and returns a list of all the
        test items contained within.
        """
        ...
    
    def runitem(self, source: str) -> Any:
        """Run the "test_func" Item.

        The calling test instance (class containing the test method) must
        provide a ``.getrunner()`` method which should return a runner which
        can run the test protocol for a single item, e.g.
        :py:func:`_pytest.runner.runtestprotocol`.
        """
        ...
    
    def inline_runsource(self, source: str, *cmdlineargs) -> HookRecorder:
        """Run a test module in process using ``pytest.main()``.

        This run writes "source" into a temporary file and runs
        ``pytest.main()`` on it, returning a :py:class:`HookRecorder` instance
        for the result.

        :param source: The source code of the test module.

        :param cmdlineargs: Any extra command line arguments to use.

        :returns: :py:class:`HookRecorder` instance of the result.
        """
        ...
    
    def inline_genitems(self, *args) -> Tuple[List[Item], HookRecorder]:
        """Run ``pytest.main(['--collectonly'])`` in-process.

        Runs the :py:func:`pytest.main` function to run all of pytest inside
        the test process itself like :py:meth:`inline_run`, but returns a
        tuple of the collected items and a :py:class:`HookRecorder` instance.
        """
        ...
    
    def inline_run(self, *args: Union[str, os.PathLike[str]], plugins=..., no_reraise_ctrlc: bool = ...) -> HookRecorder:
        """Run ``pytest.main()`` in-process, returning a HookRecorder.

        Runs the :py:func:`pytest.main` function to run all of pytest inside
        the test process itself.  This means it can return a
        :py:class:`HookRecorder` instance which gives more detailed results
        from that run than can be done by matching stdout/stderr from
        :py:meth:`runpytest`.

        :param args:
            Command line arguments to pass to :py:func:`pytest.main`.
        :param plugins:
            Extra plugin instances the ``pytest.main()`` instance should use.
        :param no_reraise_ctrlc:
            Typically we reraise keyboard interrupts from the child run. If
            True, the KeyboardInterrupt exception is captured.

        :returns: A :py:class:`HookRecorder` instance.
        """
        ...
    
    def runpytest_inprocess(self, *args: Union[str, os.PathLike[str]], **kwargs: Any) -> RunResult:
        """Return result of running pytest in-process, providing a similar
        interface to what self.runpytest() provides."""
        ...
    
    def runpytest(self, *args: Union[str, os.PathLike[str]], **kwargs: Any) -> RunResult:
        """Run pytest inline or in a subprocess, depending on the command line
        option "--runpytest" and return a :py:class:`RunResult`."""
        ...
    
    def parseconfig(self, *args: Union[str, os.PathLike[str]]) -> Config:
        """Return a new pytest Config instance from given commandline args.

        This invokes the pytest bootstrapping code in _pytest.config to create
        a new :py:class:`_pytest.core.PluginManager` and call the
        pytest_cmdline_parse hook to create a new
        :py:class:`_pytest.config.Config` instance.

        If :py:attr:`plugins` has been populated they should be plugin modules
        to be registered with the PluginManager.
        """
        ...
    
    def parseconfigure(self, *args: Union[str, os.PathLike[str]]) -> Config:
        """Return a new pytest configured Config instance.

        Returns a new :py:class:`_pytest.config.Config` instance like
        :py:meth:`parseconfig`, but also calls the pytest_configure hook.
        """
        ...
    
    def getitem(self, source: str, funcname: str = ...) -> Item:
        """Return the test item for a test function.

        Writes the source to a python file and runs pytest's collection on
        the resulting module, returning the test item for the requested
        function name.

        :param source:
            The module source.
        :param funcname:
            The name of the test function for which to return a test item.
        """
        ...
    
    def getitems(self, source: str) -> List[Item]:
        """Return all test items collected from the module.

        Writes the source to a Python file and runs pytest's collection on
        the resulting module, returning all test items contained within.
        """
        ...
    
    def getmodulecol(self, source: Union[str, Path], configargs=..., *, withinit: bool = ...): # -> Collector | Item | None:
        """Return the module collection node for ``source``.

        Writes ``source`` to a file using :py:meth:`makepyfile` and then
        runs the pytest collection on it, returning the collection node for the
        test module.

        :param source:
            The source code of the module to collect.

        :param configargs:
            Any extra arguments to pass to :py:meth:`parseconfigure`.

        :param withinit:
            Whether to also write an ``__init__.py`` file to the same
            directory to ensure it is a package.
        """
        ...
    
    def collect_by_name(self, modcol: Collector, name: str) -> Optional[Union[Item, Collector]]:
        """Return the collection node for name from the module collection.

        Searchs a module collection node for a collection node matching the
        given name.

        :param modcol: A module collection node; see :py:meth:`getmodulecol`.
        :param name: The name of the node to return.
        """
        ...
    
    def popen(self, cmdargs, stdout: Union[int, TextIO] = ..., stderr: Union[int, TextIO] = ..., stdin=..., **kw): # -> Popen[str]:
        """Invoke subprocess.Popen.

        Calls subprocess.Popen making sure the current working directory is
        in the PYTHONPATH.

        You probably want to use :py:meth:`run` instead.
        """
        ...
    
    def run(self, *cmdargs: Union[str, os.PathLike[str]], timeout: Optional[float] = ..., stdin=...) -> RunResult:
        """Run a command with arguments.

        Run a process using subprocess.Popen saving the stdout and stderr.

        :param cmdargs:
            The sequence of arguments to pass to `subprocess.Popen()`, with path-like objects
            being converted to ``str`` automatically.
        :param timeout:
            The period in seconds after which to timeout and raise
            :py:class:`Pytester.TimeoutExpired`.
        :param stdin:
            Optional standard input.  Bytes are being send, closing
            the pipe, otherwise it is passed through to ``popen``.
            Defaults to ``CLOSE_STDIN``, which translates to using a pipe
            (``subprocess.PIPE``) that gets closed.

        :rtype: RunResult
        """
        ...
    
    def runpython(self, script) -> RunResult:
        """Run a python script using sys.executable as interpreter.

        :rtype: RunResult
        """
        ...
    
    def runpython_c(self, command): # -> RunResult:
        """Run python -c "command".

        :rtype: RunResult
        """
        ...
    
    def runpytest_subprocess(self, *args, timeout: Optional[float] = ...) -> RunResult:
        """Run pytest as a subprocess with given arguments.

        Any plugins added to the :py:attr:`plugins` list will be added using the
        ``-p`` command line option.  Additionally ``--basetemp`` is used to put
        any temporary files and directories in a numbered directory prefixed
        with "runpytest-" to not conflict with the normal numbered pytest
        location for temporary files and directories.

        :param args:
            The sequence of arguments to pass to the pytest subprocess.
        :param timeout:
            The period in seconds after which to timeout and raise
            :py:class:`Pytester.TimeoutExpired`.

        :rtype: RunResult
        """
        ...
    
    def spawn_pytest(self, string: str, expect_timeout: float = ...) -> pexpect.spawn:
        """Run pytest using pexpect.

        This makes sure to use the right pytest and sets up the temporary
        directory locations.

        The pexpect child is returned.
        """
        ...
    
    def spawn(self, cmd: str, expect_timeout: float = ...) -> pexpect.spawn:
        """Run a command using pexpect.

        The pexpect child is returned.
        """
        ...
    


class LineComp:
    def __init__(self) -> None:
        ...
    
    def assert_contains_lines(self, lines2: Sequence[str]) -> None:
        """Assert that ``lines2`` are contained (linearly) in :attr:`stringio`'s value.

        Lines are matched using :func:`LineMatcher.fnmatch_lines`.
        """
        ...
    


@final
@attr.s(repr=False, str=False, init=False)
class Testdir:
    """
    Similar to :class:`Pytester`, but this class works with legacy py.path.local objects instead.

    All methods just forward to an internal :class:`Pytester` instance, converting results
    to `py.path.local` objects as necessary.
    """
    __test__ = ...
    CLOSE_STDIN = Pytester.CLOSE_STDIN
    TimeoutExpired = Pytester.TimeoutExpired
    Session = Pytester.Session
    def __init__(self, pytester: Pytester, *, _ispytest: bool = ...) -> None:
        ...
    
    @property
    def tmpdir(self) -> py.path.local:
        """Temporary directory where tests are executed."""
        ...
    
    @property
    def test_tmproot(self) -> py.path.local:
        ...
    
    @property
    def request(self): # -> FixtureRequest:
        ...
    
    @property
    def plugins(self): # -> List[str | _PluggyPlugin]:
        ...
    
    @plugins.setter
    def plugins(self, plugins): # -> None:
        ...
    
    @property
    def monkeypatch(self) -> MonkeyPatch:
        ...
    
    def make_hook_recorder(self, pluginmanager) -> HookRecorder:
        """See :meth:`Pytester.make_hook_recorder`."""
        ...
    
    def chdir(self) -> None:
        """See :meth:`Pytester.chdir`."""
        ...
    
    def finalize(self) -> None:
        """See :meth:`Pytester._finalize`."""
        ...
    
    def makefile(self, ext, *args, **kwargs) -> py.path.local:
        """See :meth:`Pytester.makefile`."""
        ...
    
    def makeconftest(self, source) -> py.path.local:
        """See :meth:`Pytester.makeconftest`."""
        ...
    
    def makeini(self, source) -> py.path.local:
        """See :meth:`Pytester.makeini`."""
        ...
    
    def getinicfg(self, source: str) -> SectionWrapper:
        """See :meth:`Pytester.getinicfg`."""
        ...
    
    def makepyprojecttoml(self, source) -> py.path.local:
        """See :meth:`Pytester.makepyprojecttoml`."""
        ...
    
    def makepyfile(self, *args, **kwargs) -> py.path.local:
        """See :meth:`Pytester.makepyfile`."""
        ...
    
    def maketxtfile(self, *args, **kwargs) -> py.path.local:
        """See :meth:`Pytester.maketxtfile`."""
        ...
    
    def syspathinsert(self, path=...) -> None:
        """See :meth:`Pytester.syspathinsert`."""
        ...
    
    def mkdir(self, name) -> py.path.local:
        """See :meth:`Pytester.mkdir`."""
        ...
    
    def mkpydir(self, name) -> py.path.local:
        """See :meth:`Pytester.mkpydir`."""
        ...
    
    def copy_example(self, name=...) -> py.path.local:
        """See :meth:`Pytester.copy_example`."""
        ...
    
    def getnode(self, config: Config, arg) -> Optional[Union[Item, Collector]]:
        """See :meth:`Pytester.getnode`."""
        ...
    
    def getpathnode(self, path): # -> Item | Collector:
        """See :meth:`Pytester.getpathnode`."""
        ...
    
    def genitems(self, colitems: List[Union[Item, Collector]]) -> List[Item]:
        """See :meth:`Pytester.genitems`."""
        ...
    
    def runitem(self, source): # -> Any:
        """See :meth:`Pytester.runitem`."""
        ...
    
    def inline_runsource(self, source, *cmdlineargs): # -> HookRecorder:
        """See :meth:`Pytester.inline_runsource`."""
        ...
    
    def inline_genitems(self, *args): # -> Tuple[List[Item], HookRecorder]:
        """See :meth:`Pytester.inline_genitems`."""
        ...
    
    def inline_run(self, *args, plugins=..., no_reraise_ctrlc: bool = ...): # -> HookRecorder:
        """See :meth:`Pytester.inline_run`."""
        ...
    
    def runpytest_inprocess(self, *args, **kwargs) -> RunResult:
        """See :meth:`Pytester.runpytest_inprocess`."""
        ...
    
    def runpytest(self, *args, **kwargs) -> RunResult:
        """See :meth:`Pytester.runpytest`."""
        ...
    
    def parseconfig(self, *args) -> Config:
        """See :meth:`Pytester.parseconfig`."""
        ...
    
    def parseconfigure(self, *args) -> Config:
        """See :meth:`Pytester.parseconfigure`."""
        ...
    
    def getitem(self, source, funcname=...): # -> Item:
        """See :meth:`Pytester.getitem`."""
        ...
    
    def getitems(self, source): # -> List[Item]:
        """See :meth:`Pytester.getitems`."""
        ...
    
    def getmodulecol(self, source, configargs=..., withinit=...): # -> Collector | Item | None:
        """See :meth:`Pytester.getmodulecol`."""
        ...
    
    def collect_by_name(self, modcol: Collector, name: str) -> Optional[Union[Item, Collector]]:
        """See :meth:`Pytester.collect_by_name`."""
        ...
    
    def popen(self, cmdargs, stdout: Union[int, TextIO] = ..., stderr: Union[int, TextIO] = ..., stdin=..., **kw): # -> Popen[str]:
        """See :meth:`Pytester.popen`."""
        ...
    
    def run(self, *cmdargs, timeout=..., stdin=...) -> RunResult:
        """See :meth:`Pytester.run`."""
        ...
    
    def runpython(self, script) -> RunResult:
        """See :meth:`Pytester.runpython`."""
        ...
    
    def runpython_c(self, command): # -> RunResult:
        """See :meth:`Pytester.runpython_c`."""
        ...
    
    def runpytest_subprocess(self, *args, timeout=...) -> RunResult:
        """See :meth:`Pytester.runpytest_subprocess`."""
        ...
    
    def spawn_pytest(self, string: str, expect_timeout: float = ...) -> pexpect.spawn:
        """See :meth:`Pytester.spawn_pytest`."""
        ...
    
    def spawn(self, cmd: str, expect_timeout: float = ...) -> pexpect.spawn:
        """See :meth:`Pytester.spawn`."""
        ...
    
    def __repr__(self) -> str:
        ...
    
    def __str__(self) -> str:
        ...
    


class LineMatcher:
    """Flexible matching of text.

    This is a convenience class to test large texts like the output of
    commands.

    The constructor takes a list of lines without their trailing newlines, i.e.
    ``text.splitlines()``.
    """
    def __init__(self, lines: List[str]) -> None:
        ...
    
    def __str__(self) -> str:
        """Return the entire original text.

        .. versionadded:: 6.2
            You can use :meth:`str` in older versions.
        """
        ...
    
    def fnmatch_lines_random(self, lines2: Sequence[str]) -> None:
        """Check lines exist in the output in any order (using :func:`python:fnmatch.fnmatch`)."""
        ...
    
    def re_match_lines_random(self, lines2: Sequence[str]) -> None:
        """Check lines exist in the output in any order (using :func:`python:re.match`)."""
        ...
    
    def get_lines_after(self, fnline: str) -> Sequence[str]:
        """Return all lines following the given line in the text.

        The given line can contain glob wildcards.
        """
        ...
    
    def fnmatch_lines(self, lines2: Sequence[str], *, consecutive: bool = ...) -> None:
        """Check lines exist in the output (using :func:`python:fnmatch.fnmatch`).

        The argument is a list of lines which have to match and can use glob
        wildcards.  If they do not match a pytest.fail() is called.  The
        matches and non-matches are also shown as part of the error message.

        :param lines2: String patterns to match.
        :param consecutive: Match lines consecutively?
        """
        ...
    
    def re_match_lines(self, lines2: Sequence[str], *, consecutive: bool = ...) -> None:
        """Check lines exist in the output (using :func:`python:re.match`).

        The argument is a list of lines which have to match using ``re.match``.
        If they do not match a pytest.fail() is called.

        The matches and non-matches are also shown as part of the error message.

        :param lines2: string patterns to match.
        :param consecutive: match lines consecutively?
        """
        ...
    
    def no_fnmatch_line(self, pat: str) -> None:
        """Ensure captured lines do not match the given pattern, using ``fnmatch.fnmatch``.

        :param str pat: The pattern to match lines.
        """
        ...
    
    def no_re_match_line(self, pat: str) -> None:
        """Ensure captured lines do not match the given pattern, using ``re.match``.

        :param str pat: The regular expression to match lines.
        """
        ...
    
    def str(self) -> str:
        """Return the entire original text."""
        ...
    


