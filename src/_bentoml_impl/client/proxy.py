from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import typing as t

from _bentoml_sdk import Service
from _bentoml_sdk.api import APIMethod
from bentoml.exceptions import BentoMLException

from .http import AbstractClient
from .http import AsyncHTTPClient
from .http import SyncHTTPClient

T = t.TypeVar("T")
logger = logging.getLogger("bentoml.io")


class RemoteProxy(AbstractClient, t.Generic[T]):
    """A remote proxy of the passed in service that has the same interfaces"""

    def __init__(
        self,
        url: str,
        *,
        service: Service[T] | None = None,
    ) -> None:
        assert service is not None, "service must be provided"
        self._sync = SyncHTTPClient(
            url, media_type="application/vnd.bentoml+pickle", service=service
        )
        self._async = AsyncHTTPClient(
            url, media_type="application/vnd.bentoml+pickle", service=service
        )
        self._inner = service.inner
        self.endpoints = self._async.endpoints
        super().__init__()

    async def is_ready(self, timeout: int | None = None) -> bool:
        return await self._async.is_ready(timeout=timeout)

    async def close(self) -> None:
        from starlette.concurrency import run_in_threadpool

        await asyncio.gather(self._async.close(), run_in_threadpool(self._sync.close))

    def as_service(self) -> T:
        return t.cast(T, self)

    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        original_func = getattr(self._inner, __name)
        if not isinstance(original_func, APIMethod):
            raise BentoMLException(f"calling non-api method {__name} is not allowed")
        original_func = original_func.func
        while isinstance(original_func, functools.partial):
            original_func = original_func.func
        is_async_func = (
            asyncio.iscoroutinefunction(original_func)
            or (
                callable(original_func)
                and asyncio.iscoroutinefunction(original_func.__call__)  # type: ignore
            )
            or inspect.isasyncgenfunction(original_func)
        )
        if is_async_func:
            return self._async.call(__name, *args, **kwargs)
        else:
            return self._sync.call(__name, *args, **kwargs)
