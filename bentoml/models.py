import typing as t

from ._internal.models.store import (  # noqa # pylint: disable
    delete,
    export,
    get,
    imports,
    ls,
    register,
)

__all__ = ["delete", "export", "imports", "get", "ls", "register"]

_T = t.TypeVar("_T")


def docstrings(docs: str):
    def decorator(func: t.Callable[..., _T]):
        func.__doc__ = docs
        return func

    return decorator
