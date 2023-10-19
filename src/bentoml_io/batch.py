from __future__ import annotations

import functools
import inspect
import typing as t
from typing import Any

from typing_extensions import get_args

from ...exceptions import ServiceUnavailable
from ..marshal.dispatcher import CorkDispatcher
from ..runner.runnable import RunnableMethod
from ..utils import first_not_none

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


class batch_infer(BatchDecorator[S, T, P, R]):
    """A decorator that makes a batch inference function callable like a normal one,
    where the input are simply stacked into a list of the same type.

    Args:
        max_batch_size (int, optional): The maximum batch size to be used for inference.
        max_latency_ms (int, optional): The maximum latency in milliseconds to be used for inference.
        batch_dim (tuple[int, int] | int, optional): The dimension of the batch. If it is a tuple, the first element
            is for input batch dimension and the second element is for output batch dimension. If it is an integer,
            both input and output batch dimension will be the same.

    Example:

        class MyRunnable(Runnable):
            @batch_infer(max_batch_size=100, max_latency_ms=1000)
            def infer(self, input: list[np.ndarray]) -> list[float]:
                return self.model.predict(input)

            @Runnable.method()
            async def predict(self, input: np.ndarray) -> float:
                return await self.infer(input)

    .. note::
        The new function returned by this decorator will become a coroutine function.
    """

    def __init__(
        self,
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        batch_dim: tuple[int, int] | int = 0,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.batch_dim = (
            batch_dim if isinstance(batch_dim, tuple) else (batch_dim, batch_dim)
        )
        self._dispatcher: CorkDispatcher[T, R, R] | None = None
        self._get_batch_size: t.Callable[[T], int] = len

    def get_batch_size(self, callback: t.Callable[[T], int]) -> t.Callable[[T], int]:
        self._get_batch_size = callback
        return callback

    def __call__(self, func: Any) -> Any:
        if isinstance(func, RunnableMethod):
            raise TypeError(
                "@batch_infer decorator should be used under @Runnable.method"
            )

        if inspect.isasyncgenfunction(func) or inspect.isgeneratorfunction(func):
            raise TypeError(
                "@batch_infer decorator cannot be used on generator function"
            )

        if hasattr(func, "get_batch_size"):
            raise TypeError("The function can't be decorated by @batch_infer twice")

        async def wrapper(runnable_self: Runnable, *args: Any, **kwargs: Any) -> R:
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

            if len(args) > 1:
                raise TypeError("Batch inference function only accept one argument")
            elif args:
                arg = args[0]
            else:
                arg_names = [k for k in kwargs if k not in ("ctx", "context")]
                if len(arg_names) != 1:
                    raise TypeError("Batch inference function only accept one argument")
                arg = kwargs.pop(arg_names[0])

            async def infer_batch(batches: t.Sequence[T]) -> t.Sequence[R]:
                from starlette.concurrency import run_in_threadpool

                from ..utils import is_async_callable

                if is_async_callable(func):
                    result = await func(runnable_self, batches, **kwargs)
                else:
                    result = await run_in_threadpool(
                        func, runnable_self, batches, **kwargs
                    )
                return result

            return await self._dispatcher(infer_batch)(arg)

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
