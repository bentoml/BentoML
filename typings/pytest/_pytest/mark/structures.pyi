
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import attr
from _pytest.config import Config
from _pytest.fixtures import _Scope

from ..compat import NotSetType, final
from ..nodes import Node

EMPTY_PARAMETERSET_OPTION: str = ...
def istestfunc(func) -> bool:
    ...

def get_empty_parameterset_mark(config: Config, argnames: Sequence[str], func) -> MarkDecorator:
    ...

class ParameterSet(NamedTuple("ParameterSet", [("values", Sequence[Union[object, NotSetType]]), ("marks", Collection[Union["MarkDecorator", "Mark"]]), ("id", Optional[str])])):
    @classmethod
    def param(cls, *values: object, marks: Union[MarkDecorator, Collection[Union[MarkDecorator, Mark]]] = ..., id: Optional[str] = ...) -> ParameterSet:
        ...

    @classmethod
    def extract_from(cls, parameterset: Union[ParameterSet, Sequence[object], object], force_tuple: bool = ...) -> ParameterSet:
        """Extract from an object or objects.

        :param parameterset:
            A legacy style parameterset that may or may not be a tuple,
            and may or may not be wrapped into a mess of mark objects.

        :param force_tuple:
            Enforce tuple wrapping so single argument tuple values
            don't get decomposed and break tests.
        """
        ...



@final
@attr.s(frozen=True)
class Mark:
    name:str = ...
    args: Tuple[Any, ...] = ...
    kwargs: Mapping[str, Any] = ...
    _param_ids_from: Optional["Mark"] = ...
    _param_ids_generated: Optional[Sequence[str]] = ...
    def combined_with(self, other: "Mark") -> "Mark":
        ...



_Markable = TypeVar("_Markable", bound=Union[Callable[..., object], type])
@attr.s
class MarkDecorator:
    """A decorator for applying a mark on test functions and classes.

    MarkDecorators are created with ``pytest.mark``::

        mark1 = pytest.mark.NAME              # Simple MarkDecorator
        mark2 = pytest.mark.NAME(name1=value) # Parametrized MarkDecorator

    and can then be applied as decorators to test functions::

        @mark2
        def test_function():
            pass

    When a MarkDecorator is called it does the following:

    1. If called with a single class as its only positional argument and no
       additional keyword arguments, it attaches the mark to the class so it
       gets applied automatically to all test cases found in that class.

    2. If called with a single function as its only positional argument and
       no additional keyword arguments, it attaches the mark to the function,
       containing all the arguments already stored internally in the
       MarkDecorator.

    3. When called in any other case, it returns a new MarkDecorator instance
       with the original MarkDecorator's content updated with the arguments
       passed to this call.

    Note: The rules above prevent MarkDecorators from storing only a single
    function or class reference as their positional argument with no
    additional keyword or positional arguments. You can work around this by
    using `with_args()`.
    """
    mark = ...
    @property
    def name(self) -> str:
        """Alias for mark.name."""
        ...

    @property
    def args(self) -> Tuple[Any, ...]:
        """Alias for mark.args."""
        ...

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Alias for mark.kwargs."""
        ...

    @property
    def markname(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def with_args(self, *args: object, **kwargs: object) -> MarkDecorator:
        """Return a MarkDecorator with extra arguments added.

        Unlike calling the MarkDecorator, with_args() can be used even
        if the sole argument is a callable/class.

        :rtype: MarkDecorator
        """
        ...

    @overload
    def __call__(self, arg: _Markable) -> _Markable:
        ...

    @overload
    def __call__(self, *args: object, **kwargs: object) -> MarkDecorator:
        ...

    def __call__(self, *args: object, **kwargs: object): # -> object | MarkDecorator:
        """Call the MarkDecorator."""
        ...



def get_unpacked_marks(obj) -> List[Mark]:
    """Obtain the unpacked marks that are stored on an object."""
    ...

def normalize_mark_list(mark_list: Iterable[Union[Mark, MarkDecorator]]) -> List[Mark]:
    """Normalize marker decorating helpers to mark objects.

    :type List[Union[Mark, Markdecorator]] mark_list:
    :rtype: List[Mark]
    """
    ...

def store_mark(obj, mark: Mark) -> None:
    """Store a Mark on an object.

    This is used to implement the Mark declarations/decorators correctly.
    """
    ...

class _SkipMarkDecorator(MarkDecorator):
    @overload
    def __call__(self, arg: _Markable) -> _Markable:
        ...

    @overload
    def __call__(self, reason: str = ...) -> MarkDecorator:
        ...



class _SkipifMarkDecorator(MarkDecorator):
    def __call__(self, condition: Union[str, bool] = ..., *conditions: Union[str, bool], reason: str = ...) -> MarkDecorator:
        ...



class _XfailMarkDecorator(MarkDecorator):
    @overload
    def __call__(self, arg: _Markable) -> _Markable:
        ...

    @overload
    def __call__(self, condition: Union[str, bool] = ..., *conditions: Union[str, bool], reason: str = ..., run: bool = ..., raises: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = ..., strict: bool = ...) -> MarkDecorator:
        ...



class _ParametrizeMarkDecorator(MarkDecorator):
    def __call__(self, argnames: Union[str, List[str], Tuple[str, ...]], argvalues: Iterable[Union[ParameterSet, Sequence[object], object]], *, indirect: Union[bool, Sequence[str]] = ..., ids: Optional[Union[Iterable[Union[None, str, float, int, bool]], Callable[[Any], Optional[object]]],] = ..., scope: Optional[_Scope] = ...) -> MarkDecorator:
        ...



class _UsefixturesMarkDecorator(MarkDecorator):
    def __call__(self, *fixtures: str) -> MarkDecorator:
        ...



class _FilterwarningsMarkDecorator(MarkDecorator):
    def __call__(self, *filters: str) -> MarkDecorator:
        ...



@final
class MarkGenerator:
    """Factory for :class:`MarkDecorator` objects - exposed as
    a ``pytest.mark`` singleton instance.

    Example::

         import pytest

         @pytest.mark.slowtest
         def test_function():
            pass

    applies a 'slowtest' :class:`Mark` on ``test_function``.
    """
    _config: Optional[Config] = ...
    _markers: Set[str] = ...
    skip: _SkipMarkDecorator
    skipif: _SkipifMarkDecorator
    xfail: _XfailMarkDecorator
    parametrize: _ParametrizeMarkDecorator
    usefixtures: _UsefixturesMarkDecorator
    filterwarnings: _FilterwarningsMarkDecorator
    ...
    def __getattr__(self, name: str) -> MarkDecorator:
        ...



MARK_GEN: MarkGenerator = ...
@final
class NodeKeywords(MutableMapping[str, Any]):
    def __init__(self, node: Node) -> None:
        ...

    def __getitem__(self, key: str) -> Any:
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        ...

    def __delitem__(self, key: str) -> None:
        ...

    def __iter__(self) -> Iterator[str]:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...



