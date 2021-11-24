
import traceback
from types import CodeType, FrameType, TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)
from weakref import ReferenceType

import attr
import py
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest.compat import final
from typing_extensions import Literal

if TYPE_CHECKING:
    _TracebackStyle = Literal["long", "short", "line", "no", "native", "value", "auto"]
class Code:
    """Wrapper around Python code objects."""
    __slots__ = ...
    def __init__(self, obj: CodeType) -> None:
        ...
    
    @classmethod
    def from_function(cls, obj: object) -> Code:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    __hash__ = ...
    @property
    def firstlineno(self) -> int:
        ...
    
    @property
    def name(self) -> str:
        ...
    
    @property
    def path(self) -> Union[py.path.local, str]:
        """Return a path object pointing to source code, or an ``str`` in
        case of ``OSError`` / non-existing file."""
        ...
    
    @property
    def fullsource(self) -> Optional[Source]:
        """Return a _pytest._code.Source object for the full source file of the code."""
        ...
    
    def source(self) -> Source:
        """Return a _pytest._code.Source object for the code object's source only."""
        ...
    
    def getargs(self, var: bool = ...) -> Tuple[str, ...]:
        """Return a tuple with the argument names for the code object.

        If 'var' is set True also return the names of the variable and
        keyword arguments when present.
        """
        ...
    


class Frame:
    """Wrapper around a Python frame holding f_locals and f_globals
    in which expressions can be evaluated."""
    __slots__ = ...
    def __init__(self, frame: FrameType) -> None:
        ...
    
    @property
    def lineno(self) -> int:
        ...
    
    @property
    def f_globals(self) -> Dict[str, Any]:
        ...
    
    @property
    def f_locals(self) -> Dict[str, Any]:
        ...
    
    @property
    def code(self) -> Code:
        ...
    
    @property
    def statement(self) -> Source:
        """Statement this frame is at."""
        ...
    
    def eval(self, code, **vars):
        """Evaluate 'code' in the frame.

        'vars' are optional additional local variables.

        Returns the result of the evaluation.
        """
        ...
    
    def repr(self, object: object) -> str:
        """Return a 'safe' (non-recursive, one-line) string repr for 'object'."""
        ...
    
    def getargs(self, var: bool = ...):
        """Return a list of tuples (name, value) for all arguments.

        If 'var' is set True, also include the variable and keyword arguments
        when present.
        """
        ...
    


class TracebackEntry:
    """A single entry in a Traceback."""
    __slots__ = ...
    def __init__(self, rawentry: TracebackType, excinfo: Optional[ReferenceType[ExceptionInfo[BaseException]]] = ...) -> None:
        ...
    
    @property
    def lineno(self) -> int:
        ...
    
    def set_repr_style(self, mode: Literal[short, long]) -> None:
        ...
    
    @property
    def frame(self) -> Frame:
        ...
    
    @property
    def relline(self) -> int:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def statement(self) -> Source:
        """_pytest._code.Source object for the current statement."""
        ...
    
    @property
    def path(self) -> Union[py.path.local, str]:
        """Path to the source code."""
        ...
    
    @property
    def locals(self) -> Dict[str, Any]:
        """Locals of underlying frame."""
        ...
    
    def getfirstlinesource(self) -> int:
        ...
    
    def getsource(self, astcache=...) -> Optional[Source]:
        """Return failing source code."""
        ...
    
    source = ...
    def ishidden(self) -> bool:
        """Return True if the current frame has a var __tracebackhide__
        resolving to True.

        If __tracebackhide__ is a callable, it gets called with the
        ExceptionInfo instance and can decide whether to hide the traceback.

        Mostly for internal use.
        """
        ...
    
    def __str__(self) -> str:
        ...
    
    @property
    def name(self) -> str:
        """co_name of underlying code."""
        ...
    


class Traceback(List[TracebackEntry]):
    """Traceback objects encapsulate and offer higher level access to Traceback entries."""
    def __init__(self, tb: Union[TracebackType, Iterable[TracebackEntry]], excinfo: Optional[ReferenceType[ExceptionInfo[BaseException]]] = ...) -> None:
        """Initialize from given python traceback object and ExceptionInfo."""
        ...
    
    def cut(self, path=..., lineno: Optional[int] = ..., firstlineno: Optional[int] = ..., excludepath: Optional[py.path.local] = ...) -> Traceback:
        """Return a Traceback instance wrapping part of this Traceback.

        By providing any combination of path, lineno and firstlineno, the
        first frame to start the to-be-returned traceback is determined.

        This allows cutting the first part of a Traceback instance e.g.
        for formatting reasons (removing some uninteresting bits that deal
        with handling of the exception/traceback).
        """
        ...
    
    @overload
    def __getitem__(self, key: int) -> TracebackEntry:
        ...
    
    @overload
    def __getitem__(self, key: slice) -> Traceback:
        ...
    
    def __getitem__(self, key: Union[int, slice]) -> Union[TracebackEntry, Traceback]:
        ...
    
    def filter(self, fn: Callable[[TracebackEntry], bool] = ...) -> Traceback:
        """Return a Traceback instance with certain items removed

        fn is a function that gets a single argument, a TracebackEntry
        instance, and should return True when the item should be added
        to the Traceback, False when not.

        By default this removes all the TracebackEntries which are hidden
        (see ishidden() above).
        """
        ...
    
    def getcrashentry(self) -> TracebackEntry:
        """Return last non-hidden traceback entry that lead to the exception of a traceback."""
        ...
    
    def recursionindex(self) -> Optional[int]:
        """Return the index of the frame/TracebackEntry where recursion originates if
        appropriate, None if no recursion occurred."""
        ...
    


co_equal = ...
_E = ...
@final
@attr.s(repr=False)
class ExceptionInfo(Generic[_E]):
    """Wraps sys.exc_info() objects and offers help for navigating the traceback."""
    _assert_start_repr = ...
    _excinfo = ...
    _striptext = ...
    _traceback = ...
    @classmethod
    def from_exc_info(cls, exc_info: Tuple[Type[_E], _E, TracebackType], exprinfo: Optional[str] = ...) -> ExceptionInfo[_E]:
        """Return an ExceptionInfo for an existing exc_info tuple.

        .. warning::

            Experimental API

        :param exprinfo:
            A text string helping to determine if we should strip
            ``AssertionError`` from the output. Defaults to the exception
            message/``__str__()``.
        """
        ...
    
    @classmethod
    def from_current(cls, exprinfo: Optional[str] = ...) -> ExceptionInfo[BaseException]:
        """Return an ExceptionInfo matching the current traceback.

        .. warning::

            Experimental API

        :param exprinfo:
            A text string helping to determine if we should strip
            ``AssertionError`` from the output. Defaults to the exception
            message/``__str__()``.
        """
        ...
    
    @classmethod
    def for_later(cls) -> ExceptionInfo[_E]:
        """Return an unfilled ExceptionInfo."""
        ...
    
    def fill_unfilled(self, exc_info: Tuple[Type[_E], _E, TracebackType]) -> None:
        """Fill an unfilled ExceptionInfo created with ``for_later()``."""
        ...
    
    @property
    def type(self) -> Type[_E]:
        """The exception class."""
        ...
    
    @property
    def value(self) -> _E:
        """The exception value."""
        ...
    
    @property
    def tb(self) -> TracebackType:
        """The exception raw traceback."""
        ...
    
    @property
    def typename(self) -> str:
        """The type name of the exception."""
        ...
    
    @property
    def traceback(self) -> Traceback:
        """The traceback."""
        ...
    
    @traceback.setter
    def traceback(self, value: Traceback) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def exconly(self, tryshort: bool = ...) -> str:
        """Return the exception as a string.

        When 'tryshort' resolves to True, and the exception is a
        _pytest._code._AssertionError, only the actual exception part of
        the exception representation is returned (so 'AssertionError: ' is
        removed from the beginning).
        """
        ...
    
    def errisinstance(self, exc: Union[Type[BaseException], Tuple[Type[BaseException], ...]]) -> bool:
        """Return True if the exception is an instance of exc.

        Consider using ``isinstance(excinfo.value, exc)`` instead.
        """
        ...
    
    def getrepr(self, showlocals: bool = ..., style: _TracebackStyle = ..., abspath: bool = ..., tbfilter: bool = ..., funcargs: bool = ..., truncate_locals: bool = ..., chain: bool = ...) -> Union[ReprExceptionInfo, ExceptionChainRepr]:
        """Return str()able representation of this exception info.

        :param bool showlocals:
            Show locals per traceback entry.
            Ignored if ``style=="native"``.

        :param str style:
            long|short|no|native|value traceback style.

        :param bool abspath:
            If paths should be changed to absolute or left unchanged.

        :param bool tbfilter:
            Hide entries that contain a local variable ``__tracebackhide__==True``.
            Ignored if ``style=="native"``.

        :param bool funcargs:
            Show fixtures ("funcargs" for legacy purposes) per traceback entry.

        :param bool truncate_locals:
            With ``showlocals==True``, make sure locals can be safely represented as strings.

        :param bool chain:
            If chained exceptions in Python 3 should be shown.

        .. versionchanged:: 3.9

            Added the ``chain`` parameter.
        """
        ...
    
    def match(self, regexp: Union[str, Pattern[str]]) -> Literal[True]:
        """Check whether the regular expression `regexp` matches the string
        representation of the exception using :func:`python:re.search`.

        If it matches `True` is returned, otherwise an `AssertionError` is raised.
        """
        ...
    


@attr.s
class FormattedExcinfo:
    """Presenting information about failing Functions and Generators."""
    flow_marker = ...
    fail_marker = ...
    showlocals = ...
    style = ...
    abspath = ...
    tbfilter = ...
    funcargs = ...
    truncate_locals = ...
    chain = ...
    astcache = ...
    def repr_args(self, entry: TracebackEntry) -> Optional[ReprFuncArgs]:
        ...
    
    def get_source(self, source: Optional[Source], line_index: int = ..., excinfo: Optional[ExceptionInfo[BaseException]] = ..., short: bool = ...) -> List[str]:
        """Return formatted and marked up source lines."""
        ...
    
    def get_exconly(self, excinfo: ExceptionInfo[BaseException], indent: int = ..., markall: bool = ...) -> List[str]:
        ...
    
    def repr_locals(self, locals: Mapping[str, object]) -> Optional[ReprLocals]:
        ...
    
    def repr_traceback_entry(self, entry: TracebackEntry, excinfo: Optional[ExceptionInfo[BaseException]] = ...) -> ReprEntry:
        ...
    
    def repr_traceback(self, excinfo: ExceptionInfo[BaseException]) -> ReprTraceback:
        ...
    
    def repr_excinfo(self, excinfo: ExceptionInfo[BaseException]) -> ExceptionChainRepr:
        ...
    


@attr.s(eq=False)
class TerminalRepr:
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


@attr.s(eq=False)
class ExceptionRepr(TerminalRepr):
    reprcrash: Optional[ReprFileLocation]
    reprtraceback: ReprTraceback
    def __attrs_post_init__(self) -> None:
        ...
    
    def addsection(self, name: str, content: str, sep: str = ...) -> None:
        ...
    
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


@attr.s(eq=False)
class ExceptionChainRepr(ExceptionRepr):
    chain = ...
    def __attrs_post_init__(self) -> None:
        ...
    
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


@attr.s(eq=False)
class ReprExceptionInfo(ExceptionRepr):
    reprtraceback = ...
    reprcrash = ...
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


@attr.s(eq=False)
class ReprTraceback(TerminalRepr):
    reprentries = ...
    extraline = ...
    style = ...
    entrysep = ...
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


class ReprTracebackNative(ReprTraceback):
    def __init__(self, tblines: Sequence[str]) -> None:
        ...
    


@attr.s(eq=False)
class ReprEntryNative(TerminalRepr):
    lines = ...
    style: _TracebackStyle = ...
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


@attr.s(eq=False)
class ReprEntry(TerminalRepr):
    lines = ...
    reprfuncargs = ...
    reprlocals = ...
    reprfileloc = ...
    style = ...
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


@attr.s(eq=False)
class ReprFileLocation(TerminalRepr):
    path = ...
    lineno = ...
    message = ...
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


@attr.s(eq=False)
class ReprLocals(TerminalRepr):
    lines = ...
    def toterminal(self, tw: TerminalWriter, indent=...) -> None:
        ...
    


@attr.s(eq=False)
class ReprFuncArgs(TerminalRepr):
    args = ...
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


def getfslineno(obj: object) -> Tuple[Union[str, py.path.local], int]:
    """Return source location (path, lineno) for the given object.

    If the source cannot be determined return ("", -1).

    The line number is 0-based.
    """
    ...

_PLUGGY_DIR = ...
if _PLUGGY_DIR.name == "__init__.py":
    _PLUGGY_DIR = ...
_PYTEST_DIR = ...
_PY_DIR = ...
def filter_traceback(entry: TracebackEntry) -> bool:
    """Return True if a TracebackEntry instance should be included in tracebacks.

    We hide traceback entries of:

    * dynamically generated code (no code to show up for it);
    * internal traceback from pytest or its internal libraries, py and pluggy.
    """
    ...

