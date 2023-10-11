from __future__ import annotations

import functools
import inspect
import typing as t
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from typing_extensions import get_args

from bentoml._internal.runner.runnable import RunnableMethod
from bentoml._internal.utils import first_not_none

from ...exceptions import ServiceUnavailable
from ..marshal.dispatcher import CorkDispatcher

if t.TYPE_CHECKING:
    from ..runner import Runnable

S = t.TypeVar("S", bound="Runnable")
T = t.TypeVar("T")
R = t.TypeVar("R")
P = t.ParamSpec("P")


class BatchDecorator(t.Protocol[S, T, P, R]):
    @t.overload
    def __call__(
        self, func: t.Callable[t.Concatenate[S, t.Sequence[T], P], t.Sequence[R]]
    ) -> t.Callable[t.Concatenate[S, T, P], t.Coroutine[None, None, R]]:
        ...

    @t.overload
    def __call__(
        self,
        func: t.Callable[
            t.Concatenate[S, t.Sequence[T], P], t.Coroutine[None, None, t.Sequence[R]]
        ],
    ) -> t.Callable[t.Concatenate[S, T, P], t.Coroutine[None, None, R]]:
        ...


def fallback() -> t.NoReturn:
    raise ServiceUnavailable("process is overloaded")


def _get_child_type(type_: t.Type[t.Any]) -> t.Type[t.Any]:
    args = get_args(type_)
    if len(args) > 0:
        return args[0]
    return t.Any


@dataclass
class batch_infer(BatchDecorator[S, T, P, R]):
    """A decorator that makes a batch inference function callable like a normal one,
    where the input are simply stacked into a list of the same type.

    Args:
        max_batch_size (int, optional): The maximum batch size to be used for inference.
        max_latency_ms (int, optional): The maximum latency in milliseconds to be used for inference.

    Example:

        class MyRunnable(Runnable):
            @batch_infer(max_batch_size=100, max_latency_ms=1000)
            def infer(self, input: list[np.ndarray]) -> list[float]:
                return self.model.predict(input)

            @Runnable.method()
            async def predict(self, input: np.ndarray) -> float:
                return await self.infer(input)
    """

    max_batch_size: int | None = None
    max_latency_ms: int | None = None
    _dispatcher: CorkDispatcher[t.Sequence[T], t.Sequence[R]] | None = field(
        default=None, init=False
    )
    _get_batch_size: t.Callable[[t.Sequence[T]], int] = field(default=len, init=False)

    def get_batch_size(
        self, callback: t.Callable[[t.Sequence[T]], int]
    ) -> t.Callable[[t.Sequence[T]], int]:
        self._get_batch_size = callback
        return callback

    def __call__(self, func: Any) -> Any:
        if isinstance(func, RunnableMethod):
            raise TypeError(
                "@batch_infer decorator should be used under @Runnable.method"
            )

        async def wrapper(runnable_self: Runnable, *args: Any, **kwargs: Any) -> Any:
            from ..container import BentoMLContainer

            if self._dispatcher is None:
                runner_config = BentoMLContainer.runners_config.get()
                max_latency_ms = first_not_none(
                    self.max_latency_ms,
                    default=runner_config["batching"]["max_latency_ms"],
                )
                max_batch_size = first_not_none(
                    self.max_batch_size,
                    default=runner_config["batching"]["max_batch_size"],
                )
                self._dispatcher = CorkDispatcher(
                    max_latency_in_ms=max_latency_ms,
                    max_batch_size=max_batch_size,
                    fallback=fallback,
                    get_batch_size=self._get_batch_size,
                )

            callback = functools.partial(func, runnable_self, **kwargs)
            infer = self._dispatcher(callback)
            return await infer(args[0])

        functools.update_wrapper(wrapper, func)
        signature = inspect.signature(func)
        params = list(signature.parameters.values())
        params[1] = params[1].replace(annotation=_get_child_type(params[1].annotation))
        wrapper.__signature__ = signature.replace(
            parameters=params,
            return_annotation=_get_child_type(signature.return_annotation),
        )
        wrapper.get_batch_size = self.get_batch_size
        return wrapper
