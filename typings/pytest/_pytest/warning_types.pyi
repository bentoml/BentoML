
from typing import Any, Generic

import attr
from _pytest.compat import final

class PytestWarning(UserWarning):
    """Base class for all warnings emitted by pytest."""
    __module__ = ...


@final
class PytestAssertRewriteWarning(PytestWarning):
    """Warning emitted by the pytest assert rewrite module."""
    __module__ = ...


@final
class PytestCacheWarning(PytestWarning):
    """Warning emitted by the cache plugin in various situations."""
    __module__ = ...


@final
class PytestConfigWarning(PytestWarning):
    """Warning emitted for configuration issues."""
    __module__ = ...


@final
class PytestCollectionWarning(PytestWarning):
    """Warning emitted when pytest is not able to collect a file or symbol in a module."""
    __module__ = ...


@final
class PytestDeprecationWarning(PytestWarning, DeprecationWarning):
    """Warning class for features that will be removed in a future version."""
    __module__ = ...


@final
class PytestExperimentalApiWarning(PytestWarning, FutureWarning):
    """Warning category used to denote experiments in pytest.

    Use sparingly as the API might change or even be removed completely in a
    future version.
    """
    __module__ = ...
    @classmethod
    def simple(cls, apiname: str) -> PytestExperimentalApiWarning:
        ...
    


@final
class PytestUnhandledCoroutineWarning(PytestWarning):
    """Warning emitted for an unhandled coroutine.

    A coroutine was encountered when collecting test functions, but was not
    handled by any async-aware plugin.
    Coroutine test functions are not natively supported.
    """
    __module__ = ...


@final
class PytestUnknownMarkWarning(PytestWarning):
    """Warning emitted on use of unknown markers.

    See :ref:`mark` for details.
    """
    __module__ = ...


@final
class PytestUnraisableExceptionWarning(PytestWarning):
    """An unraisable exception was reported.

    Unraisable exceptions are exceptions raised in :meth:`__del__ <object.__del__>`
    implementations and similar situations when the exception cannot be raised
    as normal.
    """
    __module__ = ...


@final
class PytestUnhandledThreadExceptionWarning(PytestWarning):
    """An unhandled exception occurred in a :class:`~threading.Thread`.

    Such exceptions don't propagate normally.
    """
    __module__ = ...


_W = ...
@final
@attr.s
class UnformattedWarning(Generic[_W]):
    """A warning meant to be formatted during runtime.

    This is used to hold warnings that need to format their message at runtime,
    as opposed to a direct message.
    """
    category = ...
    template = ...
    def format(self, **kwargs: Any) -> _W:
        """Return an instance of the warning category, formatted with given kwargs."""
        ...
    


