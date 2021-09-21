import typing as t

from ._internal.models.store import delete, get, ls, register  # noqa # pylint: disable

_T = t.TypeVar("_T")


def docstrings(docs: str):
    def decorator(func: t.Callable[..., _T]):
        func.__doc__ = docs
        return func

    return decorator
