from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import typing as t

from _bentoml_impl.client.base import ClientEndpoint
from _bentoml_sdk import Service
from bentoml.exceptions import BentoMLException

from .http import AbstractClient
from .http import AsyncHTTPClient
from .http import SyncHTTPClient

if t.TYPE_CHECKING:
    from _bentoml_sdk.service import ServiceConfig
    from bentoml._internal.external_typing import ASGIApp
T = t.TypeVar("T")
logger = logging.getLogger("bentoml.impl")


class RemoteProxy(AbstractClient, t.Generic[T]):
    """A remote proxy of the passed in service that has the same interfaces"""

    def __init__(
        self,
        url: str,
        *,
        service: Service[T] | None = None,
        media_type: str = "application/vnd.bentoml+pickle",
        app: ASGIApp | None = None,
    ) -> None:
        from bentoml.container import BentoMLContainer

        if service is not None:
            svc_config: dict[str, ServiceConfig] = (
                BentoMLContainer.config.services.get()
            )
            timeout = (
                svc_config.get(service.name, {}).get("traffic", {}).get("timeout") or 60
            ) * 1.01  # get the service timeout add 1% margin for the client
        else:
            timeout = 60
        self._sync = SyncHTTPClient(
            url,
            media_type=media_type,
            service=service,
            timeout=timeout,
            server_ready_timeout=0,
            app=app,
        )
        self._async = AsyncHTTPClient(
            url,
            media_type=media_type,
            service=service,
            timeout=timeout,
            server_ready_timeout=0,
            app=app,
        )
        # Setup async client with the same endpoints
        self._async.endpoints = self._sync.endpoints
        self._async._setup_endpoints()
        if service is not None:
            self._inner = service.inner
            self.endpoints = self._async.endpoints
        else:
            self.endpoints = {}
            self._inner = None
        self._setup_endpoints()

    @property
    def to_async(self) -> AsyncHTTPClient:
        return self._async

    @property
    def to_sync(self) -> SyncHTTPClient:
        return self._sync

    @property
    def client_url(self) -> str:
        return str(self._async.client.base_url)

    async def is_ready(self, timeout: int | None = None) -> bool:
        return await self._async.is_ready(timeout=timeout)

    async def close(self) -> None:
        from starlette.concurrency import run_in_threadpool

        await asyncio.gather(self._async.close(), run_in_threadpool(self._sync.close))

    def as_service(self) -> T:
        return t.cast(T, self)

    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if self._inner is None:
            raise BentoMLException(
                "The proxy is not callable when the service is not provided. Please use `.to_async` or `.to_sync` property."
            )
        original_func = getattr(self._inner, __name)
        if not hasattr(original_func, "func"):
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

    def _submit(
        self, __endpoint: ClientEndpoint, /, *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        original_func = getattr(self._inner, __endpoint.name)
        if not hasattr(original_func, "func"):
            raise BentoMLException(
                f"calling non-api method {__endpoint.name} is not allowed"
            )
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
            return self._async._submit(__endpoint, *args, **kwargs)
        else:
            return self._sync._submit(__endpoint, *args, **kwargs)
