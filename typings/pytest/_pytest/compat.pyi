
import enum
import sys
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Optional, Tuple, Union

import attr
from typing_extensions import Final

"""Python version compatibility code."""
if TYPE_CHECKING:
    ...
_T = ...
_S = ...
class NotSetType(enum.Enum):
    token = ...


NOTSET: Final = ...
if sys.version_info >= (3, 8):
    ...
else:
    ...
REGEX_TYPE = ...
def is_generator(func: object) -> bool:
    ...

def iscoroutinefunction(func: object) -> bool:
    """Return True if func is a coroutine function (a function defined with async
    def syntax, and doesn't contain yield), or a function decorated with
    @asyncio.coroutine.

    Note: copied and modified from Python 3.5's builtin couroutines.py to avoid
    importing asyncio directly, which in turns also initializes the "logging"
    module as a side-effect (see issue #8).
    """
    ...

def is_async_function(func: object) -> bool:
    """Return True if the given function seems to be an async function or
    an async generator."""
    ...

def getlocation(function, curdir: Optional[str] = ...) -> str:
    ...

def num_mock_patch_args(function) -> int:
    """Return number of arguments used up by mock arguments (if any)."""
    ...

def getfuncargnames(function: Callable[..., Any], *, name: str = ..., is_method: bool = ..., cls: Optional[type] = ...) -> Tuple[str, ...]:
    """Return the names of a function's mandatory arguments.

    Should return the names of all function arguments that:
    * Aren't bound to an instance or type as in instance or class methods.
    * Don't have default values.
    * Aren't bound with functools.partial.
    * Aren't replaced with mocks.

    The is_method and cls arguments indicate that the function should
    be treated as a bound method even though it's not unless, only in
    the case of cls, the function is a static method.

    The name parameter should be the original name in which the function was collected.
    """
    ...

if sys.version_info < (3, 7):
    ...
else:
    ...
def get_default_arg_names(function: Callable[..., Any]) -> Tuple[str, ...]:
    ...

_non_printable_ascii_translate_table = ...
STRING_TYPES = ...
def ascii_escaped(val: Union[bytes, str]) -> str:
    r"""If val is pure ASCII, return it as an str, otherwise, escape
    bytes objects into a sequence of escaped bytes:

    b'\xc3\xb4\xc5\xd6' -> r'\xc3\xb4\xc5\xd6'

    and escapes unicode objects into a sequence of escaped unicode
    ids, e.g.:

    r'4\nV\U00043efa\x0eMXWB\x1e\u3028\u15fd\xcd\U0007d944'

    Note:
       The obvious "v.decode('unicode-escape')" will return
       valid UTF-8 unicode if it finds them in bytes, but we
       want to return escaped bytes for any byte, even if they match
       a UTF-8 string.
    """
    ...

@attr.s
class _PytestWrapper:
    """Dummy wrapper around a function object for internal use only.

    Used to correctly unwrap the underlying function object when we are
    creating fixtures, because we wrap the function object ourselves with a
    decorator to issue warnings when the fixture function is called directly.
    """
    obj = ...


def get_real_func(obj): # -> (*args: Any, **kwargs: Any) -> Unknown | Any:
    """Get the real function object of the (possibly) wrapped object by
    functools.wraps or functools.partial."""
    ...

def get_real_method(obj, holder): # -> Any | (*args: Any, **kwargs: Any) -> Unknown:
    """Attempt to obtain the real function object that might be wrapping
    ``obj``, while at the same time returning a bound method to ``holder`` if
    the original object was a bound method."""
    ...

def getimfunc(func):
    ...

def safe_getattr(object: Any, name: str, default: Any) -> Any:
    """Like getattr but return default upon any Exception or any OutcomeException.

    Attribute access can potentially fail for 'evil' Python objects.
    See issue #214.
    It catches OutcomeException because of #2490 (issue #580), new outcomes
    are derived from BaseException instead of Exception (for more details
    check #2707).
    """
    ...

def safe_isclass(obj: object) -> bool:
    """Ignore any exception via isinstance on Python 3."""
    ...

if TYPE_CHECKING:
    ...
else:
    ...
if sys.version_info >= (3, 8):
    ...
else:
    ...
def assert_never(value: NoReturn) -> NoReturn:
    ...

