
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import py
from _pytest import fixtures, nodes
from _pytest._code.code import ExceptionInfo, TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import final
from _pytest.config import Config, ExitCode, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FuncFixtureInfo, _Scope
from _pytest.main import Session
from _pytest.mark import ParameterSet
from _pytest.mark.structures import Mark, MarkDecorator
from typing_extensions import Literal

"""Python test discovery, setup and run of test functions."""
if TYPE_CHECKING:
    ...
def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_cmdline_main(config: Config) -> Optional[Union[int, ExitCode]]:
    ...

def pytest_generate_tests(metafunc: Metafunc) -> None:
    ...

def pytest_configure(config: Config) -> None:
    ...

def async_warn_and_skip(nodeid: str) -> None:
    ...

@hookimpl(trylast=True)
def pytest_pyfunc_call(pyfuncitem: Function) -> Optional[object]:
    ...

def pytest_collect_file(path: py.path.local, parent: nodes.Collector) -> Optional[Module]:
    ...

def path_matches_patterns(path: py.path.local, patterns: Iterable[str]) -> bool:
    """Return whether path matches any of the patterns in the list of globs given."""
    ...

def pytest_pycollect_makemodule(path: py.path.local, parent) -> Module:
    ...

@hookimpl(trylast=True)
def pytest_pycollect_makeitem(collector: PyCollector, name: str, obj: object): # -> Any | list[Function] | None:
    ...

class PyobjMixin:
    _ALLOW_MARKERS = ...
    if TYPE_CHECKING:
        name: str = ...
        parent: Optional[nodes.Node] = ...
        own_markers: List[Mark] = ...
        def getparent(self, cls: Type[nodes._NodeType]) -> Optional[nodes._NodeType]:
            ...
        
        def listchain(self) -> List[nodes.Node]:
            ...
        
    @property
    def module(self): # -> Any | None:
        """Python module object this node was collected from (can be None)."""
        ...
    
    @property
    def cls(self): # -> Any | None:
        """Python class object this node was collected from (can be None)."""
        ...
    
    @property
    def instance(self): # -> Any | None:
        """Python instance object this node was collected from (can be None)."""
        ...
    
    @property
    def obj(self): # -> Any:
        """Underlying Python object."""
        ...
    
    @obj.setter
    def obj(self, value): # -> None:
        ...
    
    def getmodpath(self, stopatmodule: bool = ..., includemodule: bool = ...) -> str:
        """Return Python path relative to the containing module."""
        ...
    
    def reportinfo(self) -> Tuple[Union[py.path.local, str], int, str]:
        ...
    


class _EmptyClass:
    ...


IGNORED_ATTRIBUTES = ...
class PyCollector(PyobjMixin, nodes.Collector):
    def funcnamefilter(self, name: str) -> bool:
        ...
    
    def isnosetest(self, obj: object) -> bool:
        """Look for the __test__ attribute, which is applied by the
        @nose.tools.istest decorator.
        """
        ...
    
    def classnamefilter(self, name: str) -> bool:
        ...
    
    def istestfunction(self, obj: object, name: str) -> bool:
        ...
    
    def istestclass(self, obj: object, name: str) -> bool:
        ...
    
    def collect(self) -> Iterable[Union[nodes.Item, nodes.Collector]]:
        ...
    


class Module(nodes.File, PyCollector):
    """Collector for test classes and functions."""
    def collect(self) -> Iterable[Union[nodes.Item, nodes.Collector]]:
        ...
    


class Package(Module):
    def __init__(self, fspath: py.path.local, parent: nodes.Collector, config=..., session=..., nodeid=...) -> None:
        ...
    
    def setup(self) -> None:
        ...
    
    def gethookproxy(self, fspath: py.path.local): # -> FSHookProxy:
        ...
    
    def isinitpath(self, path: py.path.local) -> bool:
        ...
    
    def collect(self) -> Iterable[Union[nodes.Item, nodes.Collector]]:
        ...
    


class Class(PyCollector):
    """Collector for test methods."""
    @classmethod
    def from_parent(cls, parent, *, name, obj=...): # -> Any:
        """The public constructor."""
        ...
    
    def collect(self) -> Iterable[Union[nodes.Item, nodes.Collector]]:
        ...
    


class Instance(PyCollector):
    _ALLOW_MARKERS = ...
    def collect(self) -> Iterable[Union[nodes.Item, nodes.Collector]]:
        ...
    
    def newinstance(self):
        ...
    


def hasinit(obj: object) -> bool:
    ...

def hasnew(obj: object) -> bool:
    ...

@final
class CallSpec2:
    def __init__(self, metafunc: Metafunc) -> None:
        ...
    
    def copy(self) -> CallSpec2:
        ...
    
    def getparam(self, name: str) -> object:
        ...
    
    @property
    def id(self) -> str:
        ...
    
    def setmulti2(self, valtypes: Mapping[str, Literal[params, funcargs]], argnames: Sequence[str], valset: Iterable[object], id: str, marks: Iterable[Union[Mark, MarkDecorator]], scopenum: int, param_index: int) -> None:
        ...
    


@final
class Metafunc:
    """Objects passed to the :func:`pytest_generate_tests <_pytest.hookspec.pytest_generate_tests>` hook.

    They help to inspect a test function and to generate tests according to
    test configuration or values specified in the class or module where a
    test function is defined.
    """
    def __init__(self, definition: FunctionDefinition, fixtureinfo: fixtures.FuncFixtureInfo, config: Config, cls=..., module=...) -> None:
        ...
    
    def parametrize(self, argnames: Union[str, List[str], Tuple[str, ...]], argvalues: Iterable[Union[ParameterSet, Sequence[object], object]], indirect: Union[bool, Sequence[str]] = ..., ids: Optional[Union[Iterable[Union[None, str, float, int, bool]], Callable[[Any], Optional[object]]],] = ..., scope: Optional[_Scope] = ..., *, _param_mark: Optional[Mark] = ...) -> None:
        """Add new invocations to the underlying test function using the list
        of argvalues for the given argnames.  Parametrization is performed
        during the collection phase.  If you need to setup expensive resources
        see about setting indirect to do it rather at test setup time.

        :param argnames:
            A comma-separated string denoting one or more argument names, or
            a list/tuple of argument strings.

        :param argvalues:
            The list of argvalues determines how often a test is invoked with
            different argument values.

            If only one argname was specified argvalues is a list of values.
            If N argnames were specified, argvalues must be a list of
            N-tuples, where each tuple-element specifies a value for its
            respective argname.

        :param indirect:
            A list of arguments' names (subset of argnames) or a boolean.
            If True the list contains all names from the argnames. Each
            argvalue corresponding to an argname in this list will
            be passed as request.param to its respective argname fixture
            function so that it can perform more expensive setups during the
            setup phase of a test rather than at collection time.

        :param ids:
            Sequence of (or generator for) ids for ``argvalues``,
            or a callable to return part of the id for each argvalue.

            With sequences (and generators like ``itertools.count()``) the
            returned ids should be of type ``string``, ``int``, ``float``,
            ``bool``, or ``None``.
            They are mapped to the corresponding index in ``argvalues``.
            ``None`` means to use the auto-generated id.

            If it is a callable it will be called for each entry in
            ``argvalues``, and the return value is used as part of the
            auto-generated id for the whole set (where parts are joined with
            dashes ("-")).
            This is useful to provide more specific ids for certain items, e.g.
            dates.  Returning ``None`` will use an auto-generated id.

            If no ids are provided they will be generated automatically from
            the argvalues.

        :param scope:
            If specified it denotes the scope of the parameters.
            The scope is used for grouping tests by parameter instances.
            It will also override any fixture-function defined scope, allowing
            to set a dynamic scope using test context or configuration.
        """
        ...
    


def idmaker(argnames: Iterable[str], parametersets: Iterable[ParameterSet], idfn: Optional[Callable[[Any], Optional[object]]] = ..., ids: Optional[List[Union[None, str]]] = ..., config: Optional[Config] = ..., nodeid: Optional[str] = ...) -> List[str]:
    ...

def show_fixtures_per_test(config): # -> int | ExitCode:
    ...

def showfixtures(config: Config) -> Union[int, ExitCode]:
    ...

def write_docstring(tw: TerminalWriter, doc: str, indent: str = ...) -> None:
    ...

class Function(PyobjMixin, nodes.Item):
    """An Item responsible for setting up and executing a Python test function.

    param name:
        The full function name, including any decorations like those
        added by parametrization (``my_func[my_param]``).
    param parent:
        The parent Node.
    param config:
        The pytest Config object.
    param callspec:
        If given, this is function has been parametrized and the callspec contains
        meta information about the parametrization.
    param callobj:
        If given, the object which will be called when the Function is invoked,
        otherwise the callobj will be obtained from ``parent`` using ``originalname``.
    param keywords:
        Keywords bound to the function object for "-k" matching.
    param session:
        The pytest Session object.
    param fixtureinfo:
        Fixture information already resolved at this fixture node..
    param originalname:
        The attribute name to use for accessing the underlying function object.
        Defaults to ``name``. Set this if name is different from the original name,
        for example when it contains decorations like those added by parametrization
        (``my_func[my_param]``).
    """
    _ALLOW_MARKERS = ...
    def __init__(self, name: str, parent, config: Optional[Config] = ..., callspec: Optional[CallSpec2] = ..., callobj=..., keywords=..., session: Optional[Session] = ..., fixtureinfo: Optional[FuncFixtureInfo] = ..., originalname: Optional[str] = ...) -> None:
        ...
    
    @classmethod
    def from_parent(cls, parent, **kw): # -> Any:
        """The public constructor."""
        ...
    
    @property
    def function(self): # -> Any:
        """Underlying python 'function' object."""
        ...
    
    def runtest(self) -> None:
        """Execute the underlying test function."""
        ...
    
    def setup(self) -> None:
        ...
    
    def repr_failure(self, excinfo: ExceptionInfo[BaseException]) -> Union[str, TerminalRepr]:
        ...
    


class FunctionDefinition(Function):
    """
    This class is a step gap solution until we evolve to have actual function definition nodes
    and manage to get rid of ``metafunc``.
    """
    def runtest(self) -> None:
        ...
    
    setup = ...


