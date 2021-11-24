
import warnings
from types import TracebackType
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    List,
    Optional,
    Pattern,
    Tuple,
    Type,
    Union,
    overload,
)

from _pytest.compat import final
from _pytest.fixtures import fixture

"""Record warnings during test function execution."""
T = ...
@fixture
def recwarn() -> Generator[WarningsRecorder, None, None]:
    """Return a :class:`WarningsRecorder` instance that records all warnings emitted by test functions.

    See http://docs.python.org/library/warnings.html for information
    on warning categories.
    """
    ...

@overload
def deprecated_call(*, match: Optional[Union[str, Pattern[str]]] = ...) -> WarningsRecorder:
    ...

@overload
def deprecated_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    ...

def deprecated_call(func: Optional[Callable[..., Any]] = ..., *args: Any, **kwargs: Any) -> Union[WarningsRecorder, Any]:
    """Assert that code produces a ``DeprecationWarning`` or ``PendingDeprecationWarning``.

    This function can be used as a context manager::

        >>> import warnings
        >>> def api_call_v2():
        ...     warnings.warn('use v3 of this api', DeprecationWarning)
        ...     return 200

        >>> import pytest
        >>> with pytest.deprecated_call():
        ...    assert api_call_v2() == 200

    It can also be used by passing a function and ``*args`` and ``**kwargs``,
    in which case it will ensure calling ``func(*args, **kwargs)`` produces one of
    the warnings types above. The return value is the return value of the function.

    In the context manager form you may use the keyword argument ``match`` to assert
    that the warning matches a text or regex.

    The context manager produces a list of :class:`warnings.WarningMessage` objects,
    one for each warning raised.
    """
    ...

@overload
def warns(expected_warning: Optional[Union[Type[Warning], Tuple[Type[Warning], ...]]], *, match: Optional[Union[str, Pattern[str]]] = ...) -> WarningsChecker:
    ...

@overload
def warns(expected_warning: Optional[Union[Type[Warning], Tuple[Type[Warning], ...]]], func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    ...

def warns(expected_warning: Optional[Union[Type[Warning], Tuple[Type[Warning], ...]]], *args: Any, match: Optional[Union[str, Pattern[str]]] = ..., **kwargs: Any) -> Union[WarningsChecker, Any]:
    r"""Assert that code raises a particular class of warning.

    Specifically, the parameter ``expected_warning`` can be a warning class or
    sequence of warning classes, and the inside the ``with`` block must issue a warning of that class or
    classes.

    This helper produces a list of :class:`warnings.WarningMessage` objects,
    one for each warning raised.

    This function can be used as a context manager, or any of the other ways
    :func:`pytest.raises` can be used::

        >>> import pytest
        >>> with pytest.warns(RuntimeWarning):
        ...    warnings.warn("my warning", RuntimeWarning)

    In the context manager form you may use the keyword argument ``match`` to assert
    that the warning matches a text or regex::

        >>> with pytest.warns(UserWarning, match='must be 0 or None'):
        ...     warnings.warn("value must be 0 or None", UserWarning)

        >>> with pytest.warns(UserWarning, match=r'must be \d+$'):
        ...     warnings.warn("value must be 42", UserWarning)

        >>> with pytest.warns(UserWarning, match=r'must be \d+$'):
        ...     warnings.warn("this is not here", UserWarning)
        Traceback (most recent call last):
          ...
        Failed: DID NOT WARN. No warnings of type ...UserWarning... was emitted...

    """
    ...

class WarningsRecorder(warnings.catch_warnings):
    """A context manager to record raised warnings.

    Adapted from `warnings.catch_warnings`.
    """
    def __init__(self, *, _ispytest: bool = ...) -> None:
        ...
    
    @property
    def list(self) -> List[warnings.WarningMessage]:
        """The list of recorded warnings."""
        ...
    
    def __getitem__(self, i: int) -> warnings.WarningMessage:
        """Get a recorded warning by index."""
        ...
    
    def __iter__(self) -> Iterator[warnings.WarningMessage]:
        """Iterate through the recorded warnings."""
        ...
    
    def __len__(self) -> int:
        """The number of recorded warnings."""
        ...
    
    def pop(self, cls: Type[Warning] = ...) -> warnings.WarningMessage:
        """Pop the first recorded warning, raise exception if not exists."""
        ...
    
    def clear(self) -> None:
        """Clear the list of recorded warnings."""
        ...
    
    def __enter__(self) -> WarningsRecorder:
        ...
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        ...
    


@final
class WarningsChecker(WarningsRecorder):
    def __init__(self, expected_warning: Optional[Union[Type[Warning], Tuple[Type[Warning], ...]]] = ..., match_expr: Optional[Union[str, Pattern[str]]] = ..., *, _ispytest: bool = ...) -> None:
        ...
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        ...
    


