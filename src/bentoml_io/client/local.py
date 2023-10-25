from __future__ import annotations

import asyncio
import inspect
import typing as t
from functools import partial

from .base import AbstractClient

if t.TYPE_CHECKING:
    from ..server.service import Service

T = t.TypeVar("T")


class LocalClient(AbstractClient):
    def __init__(self, service: Service) -> None:
        self.service = service
        self.servable = service.init_servable()


class SyncLocalClient(LocalClient):
    def __init__(self, service: Service):
        super().__init__(service)
        for name in self.service.service_methods:
            setattr(self, name, partial(self.call, name))

    def call(self, name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if name not in self.service.service_methods:
            raise ValueError(f"Method {name} not found")
        result = getattr(self.servable, name)(*args, **kwargs)
        if inspect.iscoroutine(result):
            return asyncio.run(result)
        elif inspect.isasyncgen(result):
            from bentoml._internal.utils import async_gen_to_sync

            return async_gen_to_sync(result)
        return result

    def __enter__(self: T) -> T:
        return self


class AsyncLocalClient(LocalClient):
    def __init__(self, service: Service):
        super().__init__(service)
        for name in self.service.service_methods:
            setattr(self, name, partial(self.call, name))

    def call(self, name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        from starlette.concurrency import run_in_threadpool

        from bentoml._internal.utils import is_async_callable
        from bentoml._internal.utils import sync_gen_to_async

        meth = getattr(self.servable, name)
        if inspect.isgeneratorfunction(meth):
            return sync_gen_to_async(meth(*args, **kwargs))
        elif not is_async_callable(meth) and not inspect.isasyncgenfunction(meth):
            return run_in_threadpool(meth, *args, **kwargs)
        return meth(*args, **kwargs)
