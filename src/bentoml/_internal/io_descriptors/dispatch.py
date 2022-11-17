"""
This module is inspired by cattrs.dispatch and attrs._make.
"""
from __future__ import annotations

import typing as t
from functools import lru_cache

import attr

from ..types import LazyType

if t.TYPE_CHECKING:

    F = t.Callable[..., t.Any]
    CanHandleF = t.Callable[[t.Any], bool]


@attr.define(slots=True)
class FunctionDispatch:
    """
    FunctionDispatch mimics the behaviour of functools.singledispatch, but instead dispatch
    based on the functions that can takes the type of given input type for conversion function.
    This is similar to the one from cattrs.dispatch.FunctionDispatch.
    """

    _dispatchers: list[tuple[CanHandleF, F]] = attr.field(factory=list)

    def register(self, handler: CanHandleF, func: F):
        self._dispatchers.insert(0, (handler, func))

    def dispatch(self, type_: type[t.Any]) -> F:
        for handler, func in self._dispatchers:
            try:
                can_handle = handler(type_)
            except Exception:  # pylint: disable=broad-except
                # handler method can raise any exception
                continue
            if can_handle:
                return func
        raise NotImplementedError(f"No handler for type {type_}")

    def __len__(self) -> int:
        return len(self._dispatchers)


class MultiDispatch:
    """
    Dispatching based on the type of the input. Returns a tuple of (function, constructor)
    constructor by default is None.
    """

    __slots__ = ("_function_dispatch", "_direct_dispatch", "dispatch")

    def __init__(self, fallback_fn: F):
        self._direct_dispatch: dict[type[t.Any] | LazyType[t.Any], F] = {}
        self._function_dispatch = FunctionDispatch()
        self._function_dispatch.register(lambda _: True, fallback_fn)
        self.dispatch = lru_cache(maxsize=None)(self._dispatch)

    def _dispatch(self, type_: type[t.Any]):
        direct_dispatch = self._direct_dispatch.get(type_)
        if direct_dispatch is not None:
            return direct_dispatch
        # NOTE: We need to resolve lazy types import here for dispatching.
        for ltype in filter(lambda x: isinstance(x, LazyType), self._direct_dispatch):
            if issubclass(type_, ltype.get_class()):
                return self._direct_dispatch[ltype]

        return self._function_dispatch.dispatch(type_)

    def register_direct(self, type_: type[t.Any] | LazyType[t.Any], func: F):
        self._direct_dispatch[type_] = func
        self.dispatch.cache_clear()

    def register_predicate(self, predicate: CanHandleF, fn: F):
        self._function_dispatch.register(predicate, fn)
        self._direct_dispatch.clear()
        self.dispatch.cache_clear()

    def register_iter_fns(self, predicates: t.Iterable[tuple[CanHandleF, F]]):
        for pred in predicates:
            handler, fn = pred
            self._function_dispatch.register(handler, fn)
        self._direct_dispatch.clear()
        self.dispatch.cache_clear()

    def __bool__(self) -> bool:
        return len(self._function_dispatch) > 0 or len(self._direct_dispatch) > 0
