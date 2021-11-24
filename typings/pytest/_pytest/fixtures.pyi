
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

import attr
import py
from _pytest import nodes
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import final
from _pytest.config import Config, _PluggyPlugin
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.mark.structures import MarkDecorator
from _pytest.python import Function, Metafunc
from typing_extensions import Literal

if TYPE_CHECKING:
    _Scope = Literal["session", "package", "module", "class", "function"]
_FixtureValue = ...
_FixtureFunction = ...
_FixtureFunc = Union[Callable[..., _FixtureValue], Callable[..., Generator[_FixtureValue, None, None]]]
_FixtureCachedResult = Union[Tuple[_FixtureValue, object, None],, Tuple[None, object, Tuple[Type[BaseException], BaseException, TracebackType]],],
@attr.s(frozen=True)
class PseudoFixtureDef(Generic[_FixtureValue]):
    cached_result = ...
    scope = ...


def pytest_sessionstart(session: Session) -> None:
    ...

def get_scope_package(node, fixturedef: FixtureDef[object]):
    ...

def get_scope_node(node: nodes.Node, scope: _Scope) -> Optional[Union[nodes.Item, nodes.Collector]]:
    ...

name2pseudofixturedef_key = ...
def add_funcarg_pseudo_fixture_def(collector: nodes.Collector, metafunc: Metafunc, fixturemanager: FixtureManager) -> None:
    ...

def getfixturemarker(obj: object) -> Optional[FixtureFunctionMarker]:
    """Return fixturemarker or None if it doesn't exist or raised
    exceptions."""
    ...

_Key = Tuple[object, ...]
def get_parametrized_fixture_keys(item: nodes.Item, scopenum: int) -> Iterator[_Key]:
    """Return list of keys for all parametrized arguments which match
    the specified scope. """
    ...

def reorder_items(items: Sequence[nodes.Item]) -> List[nodes.Item]:
    ...

def fix_cache_order(item: nodes.Item, argkeys_cache: Dict[int, Dict[nodes.Item, Dict[_Key, None]]], items_by_argkey: Dict[int, Dict[_Key, Deque[nodes.Item]]]) -> None:
    ...

def reorder_items_atscope(items: Dict[nodes.Item, None], argkeys_cache: Dict[int, Dict[nodes.Item, Dict[_Key, None]]], items_by_argkey: Dict[int, Dict[_Key, Deque[nodes.Item]]], scopenum: int) -> Dict[nodes.Item, None]:
    ...

def fillfixtures(function: Function) -> None:
    """Fill missing fixtures for a test function (deprecated)."""
    ...

def get_direct_param_fixture_func(request):
    ...

@attr.s(slots=True)
class FuncFixtureInfo:
    argnames = ...
    initialnames = ...
    names_closure = ...
    name2fixturedefs = ...
    def prune_dependency_tree(self) -> None:
        """Recompute names_closure from initialnames and name2fixturedefs.

        Can only reduce names_closure, which means that the new closure will
        always be a subset of the old one. The order is preserved.

        This method is needed because direct parametrization may shadow some
        of the fixtures that were included in the originally built dependency
        tree. In this way the dependency tree can get pruned, and the closure
        of argnames may get reduced.
        """
        ...
    


class FixtureRequest:
    """A request for a fixture from a test or fixture function.

    A request object gives access to the requesting test context and has
    an optional ``param`` attribute in case the fixture is parametrized
    indirectly.
    """
    def __init__(self, pyfuncitem, *, _ispytest: bool = ...) -> None:
        ...
    
    @property
    def fixturenames(self) -> List[str]:
        """Names of all active fixtures in this request."""
        ...
    
    @property
    def node(self): # -> Item | Collector:
        """Underlying collection node (depends on current request scope)."""
        ...
    
    @property
    def config(self) -> Config:
        """The pytest config object associated with this request."""
        ...
    
    @property
    def function(self):
        """Test function object if the request has a per-function scope."""
        ...
    
    @property
    def cls(self): # -> None:
        """Class (can be None) where the test function was collected."""
        ...
    
    @property
    def instance(self): # -> Any | None:
        """Instance (can be None) on which test function was collected."""
        ...
    
    @property
    def module(self):
        """Python module object where the test function was collected."""
        ...
    
    @property
    def fspath(self) -> py.path.local:
        """The file system path of the test module which collected this test."""
        ...
    
    @property
    def keywords(self): # -> NodeKeywords:
        """Keywords/markers dictionary for the underlying node."""
        ...
    
    @property
    def session(self) -> Session:
        """Pytest session object."""
        ...
    
    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        """Add finalizer/teardown function to be called after the last test
        within the requesting test context finished execution."""
        ...
    
    def applymarker(self, marker: Union[str, MarkDecorator]) -> None:
        """Apply a marker to a single test function invocation.

        This method is useful if you don't want to have a keyword/marker
        on all function invocations.

        :param marker:
            A :py:class:`_pytest.mark.MarkDecorator` object created by a call
            to ``pytest.mark.NAME(...)``.
        """
        ...
    
    def raiseerror(self, msg: Optional[str]) -> NoReturn:
        """Raise a FixtureLookupError with the given message."""
        ...
    
    def getfixturevalue(self, argname: str) -> Any:
        """Dynamically run a named fixture function.

        Declaring fixtures via function argument is recommended where possible.
        But if you can only decide whether to use another fixture at test
        setup time, you may use this function to retrieve it inside a fixture
        or test function body.

        :raises pytest.FixtureLookupError:
            If the given fixture could not be found.
        """
        ...
    
    def __repr__(self) -> str:
        ...
    


@final
class SubRequest(FixtureRequest):
    """A sub request for handling getting a fixture from a test function/fixture."""
    def __init__(self, request: FixtureRequest, scope: _Scope, param, param_index: int, fixturedef: FixtureDef[object], *, _ispytest: bool = ...) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        """Add finalizer/teardown function to be called after the last test
        within the requesting test context finished execution."""
        ...
    


scopes: List[_Scope] = ...
scopenum_function = ...
def scopemismatch(currentscope: _Scope, newscope: _Scope) -> bool:
    ...

def scope2index(scope: str, descr: str, where: Optional[str] = ...) -> int:
    """Look up the index of ``scope`` and raise a descriptive value error
    if not defined."""
    ...

@final
class FixtureLookupError(LookupError):
    """Could not return a requested fixture (missing or invalid)."""
    def __init__(self, argname: Optional[str], request: FixtureRequest, msg: Optional[str] = ...) -> None:
        ...
    
    def formatrepr(self) -> FixtureLookupErrorRepr:
        ...
    


class FixtureLookupErrorRepr(TerminalRepr):
    def __init__(self, filename: Union[str, py.path.local], firstlineno: int, tblines: Sequence[str], errorstring: str, argname: Optional[str]) -> None:
        ...
    
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


def fail_fixturefunc(fixturefunc, msg: str) -> NoReturn:
    ...

def call_fixture_func(fixturefunc: _FixtureFunc[_FixtureValue], request: FixtureRequest, kwargs) -> _FixtureValue:
    ...

@final
class FixtureDef(Generic[_FixtureValue]):
    """A container for a factory definition."""
    def __init__(self, fixturemanager: FixtureManager, baseid: Optional[str], argname: str, func: _FixtureFunc[_FixtureValue], scope: Union[_Scope, Callable[[str, Config], _Scope]], params: Optional[Sequence[object]], unittest: bool = ..., ids: Optional[Union[Tuple[Union[None, str, float, int, bool], ...], Callable[[Any], Optional[object]]],] = ...) -> None:
        ...
    
    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        ...
    
    def finish(self, request: SubRequest) -> None:
        ...
    
    def execute(self, request: SubRequest) -> _FixtureValue:
        ...
    
    def cache_key(self, request: SubRequest) -> object:
        ...
    
    def __repr__(self) -> str:
        ...
    


def resolve_fixture_function(fixturedef: FixtureDef[_FixtureValue], request: FixtureRequest) -> _FixtureFunc[_FixtureValue]:
    """Get the actual callable that can be called to obtain the fixture
    value, dealing with unittest-specific instances and bound methods."""
    ...

def pytest_fixture_setup(fixturedef: FixtureDef[_FixtureValue], request: SubRequest) -> _FixtureValue:
    """Execution of fixture setup."""
    ...

def wrap_function_to_error_out_if_called_directly(function: _FixtureFunction, fixture_marker: FixtureFunctionMarker) -> _FixtureFunction:
    """Wrap the given fixture function so we can raise an error about it being called directly,
    instead of used as an argument in a test function."""
    ...

@final
@attr.s(frozen=True)
class FixtureFunctionMarker:
    scope = ...
    params = ...
    autouse = ...
    ids = ...
    name = ...
    def __call__(self, function: _FixtureFunction) -> _FixtureFunction:
        ...
    


@overload
def fixture(fixture_function: _FixtureFunction, *, scope: Union[_Scope, Callable[[str, Config], _Scope]] = ..., params: Optional[Iterable[object]] = ..., autouse: bool = ..., ids: Optional[Union[Iterable[Union[None, str, float, int, bool]], Callable[[Any], Optional[object]]],] = ..., name: Optional[str] = ...) -> _FixtureFunction:
    ...

@overload
def fixture(fixture_function: None = ..., *, scope: Union[_Scope, Callable[[str, Config], _Scope]] = ..., params: Optional[Iterable[object]] = ..., autouse: bool = ..., ids: Optional[Union[Iterable[Union[None, str, float, int, bool]], Callable[[Any], Optional[object]]],] = ..., name: Optional[str] = ...) -> FixtureFunctionMarker:
    ...

def fixture(fixture_function: Optional[_FixtureFunction] = ..., *, scope: Union[_Scope, Callable[[str, Config], _Scope]] = ..., params: Optional[Iterable[object]] = ..., autouse: bool = ..., ids: Optional[Union[Iterable[Union[None, str, float, int, bool]], Callable[[Any], Optional[object]]],] = ..., name: Optional[str] = ...) -> Union[FixtureFunctionMarker, _FixtureFunction]:
    """Decorator to mark a fixture factory function.

    This decorator can be used, with or without parameters, to define a
    fixture function.

    The name of the fixture function can later be referenced to cause its
    invocation ahead of running tests: test modules or classes can use the
    ``pytest.mark.usefixtures(fixturename)`` marker.

    Test functions can directly use fixture names as input arguments in which
    case the fixture instance returned from the fixture function will be
    injected.

    Fixtures can provide their values to test functions using ``return`` or
    ``yield`` statements. When using ``yield`` the code block after the
    ``yield`` statement is executed as teardown code regardless of the test
    outcome, and must yield exactly once.

    :param scope:
        The scope for which this fixture is shared; one of ``"function"``
        (default), ``"class"``, ``"module"``, ``"package"`` or ``"session"``.

        This parameter may also be a callable which receives ``(fixture_name, config)``
        as parameters, and must return a ``str`` with one of the values mentioned above.

        See :ref:`dynamic scope` in the docs for more information.

    :param params:
        An optional list of parameters which will cause multiple invocations
        of the fixture function and all of the tests using it. The current
        parameter is available in ``request.param``.

    :param autouse:
        If True, the fixture func is activated for all tests that can see it.
        If False (the default), an explicit reference is needed to activate
        the fixture.

    :param ids:
        List of string ids each corresponding to the params so that they are
        part of the test id. If no ids are provided they will be generated
        automatically from the params.

    :param name:
        The name of the fixture. This defaults to the name of the decorated
        function. If a fixture is used in the same module in which it is
        defined, the function name of the fixture will be shadowed by the
        function arg that requests the fixture; one way to resolve this is to
        name the decorated function ``fixture_<fixturename>`` and then use
        ``@pytest.fixture(name='<fixturename>')``.
    """
    ...

def yield_fixture(fixture_function=..., *args, scope=..., params=..., autouse=..., ids=..., name=...):
    """(Return a) decorator to mark a yield-fixture factory function.

    .. deprecated:: 3.0
        Use :py:func:`pytest.fixture` directly instead.
    """
    ...

@fixture(scope="session")
def pytestconfig(request: FixtureRequest) -> Config:
    """Session-scoped fixture that returns the :class:`_pytest.config.Config` object.

    Example::

        def test_foo(pytestconfig):
            if pytestconfig.getoption("verbose") > 0:
                ...

    """
    ...

def pytest_addoption(parser: Parser) -> None:
    ...

class FixtureManager:
    """pytest fixture definitions and information is stored and managed
    from this class.

    During collection fm.parsefactories() is called multiple times to parse
    fixture function definitions into FixtureDef objects and internal
    data structures.

    During collection of test functions, metafunc-mechanics instantiate
    a FuncFixtureInfo object which is cached per node/func-name.
    This FuncFixtureInfo object is later retrieved by Function nodes
    which themselves offer a fixturenames attribute.

    The FuncFixtureInfo object holds information about fixtures and FixtureDefs
    relevant for a particular function. An initial list of fixtures is
    assembled like this:

    - ini-defined usefixtures
    - autouse-marked fixtures along the collection chain up from the function
    - usefixtures markers at module/class/function level
    - test function funcargs

    Subsequently the funcfixtureinfo.fixturenames attribute is computed
    as the closure of the fixtures needed to setup the initial fixtures,
    i.e. fixtures needed by fixture functions themselves are appended
    to the fixturenames list.

    Upon the test-setup phases all fixturenames are instantiated, retrieved
    by a lookup of their FuncFixtureInfo.
    """
    FixtureLookupError = FixtureLookupError
    FixtureLookupErrorRepr = FixtureLookupErrorRepr
    def __init__(self, session: Session) -> None:
        ...
    
    def getfixtureinfo(self, node: nodes.Node, func, cls, funcargs: bool = ...) -> FuncFixtureInfo:
        ...
    
    def pytest_plugin_registered(self, plugin: _PluggyPlugin) -> None:
        ...
    
    def getfixtureclosure(self, fixturenames: Tuple[str, ...], parentnode: nodes.Node, ignore_args: Sequence[str] = ...) -> Tuple[Tuple[str, ...], List[str], Dict[str, Sequence[FixtureDef[Any]]]]:
        ...
    
    def pytest_generate_tests(self, metafunc: Metafunc) -> None:
        """Generate new tests based on parametrized fixtures used by the given metafunc"""
        ...
    
    def pytest_collection_modifyitems(self, items: List[nodes.Item]) -> None:
        ...
    
    def parsefactories(self, node_or_obj, nodeid=..., unittest: bool = ...) -> None:
        ...
    
    def getfixturedefs(self, argname: str, nodeid: str) -> Optional[Sequence[FixtureDef[Any]]]:
        """Get a list of fixtures which are applicable to the given node id.

        :param str argname: Name of the fixture to search for.
        :param str nodeid: Full node id of the requesting test.
        :rtype: Sequence[FixtureDef]
        """
        ...
    


