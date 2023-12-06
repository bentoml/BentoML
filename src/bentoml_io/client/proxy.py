from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import typing as t

from bentoml._internal.utils import async_gen_to_sync
from bentoml.exceptions import BentoMLException
from bentoml_io.factory import Service

from .http import ClientEndpoint
from .http import HTTPClient

T = t.TypeVar("T")
logger = logging.getLogger("bentoml.io")


class RemoteProxy(HTTPClient, t.Generic[T]):
    """A remote proxy of the passed in service that has the same interfaces"""

    def __init__(
        self,
        url: str,
        *,
        media_type: str = "application/json",
        service: Service[T] | None = None,
    ) -> None:
        assert service is not None, "service must be provided"
        super().__init__(url, media_type=media_type, service=service)
        self._inner = service.inner

    def as_service(self) -> T:
        return t.cast(T, self)

    async def is_ready(
        self, timeout: int | None = None, headers: dict[str, str] | None = None
    ) -> bool:
        import aiohttp

        client = await self._get_client()
        request_params: dict[str, t.Any] = {"headers": headers}
        if timeout is not None:
            request_params["timeout"] = aiohttp.ClientTimeout(total=timeout)
        try:
            async with client.get("/readyz", **request_params) as resp:
                return resp.status == 200
        except asyncio.TimeoutError:
            logger.warn("Timed out waiting for runner to be ready")
            return False

    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        try:
            endpoint = self.endpoints[__name]
        except KeyError:
            raise BentoMLException(f"Endpoint {__name} not found") from None
        original_func = getattr(self._inner, __name)
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
            if endpoint.stream_output:
                return self._get_async_stream(endpoint, *args, **kwargs)
            else:
                return self._call(__name, args, kwargs)
        else:
            return self._sync_call(__name, *args, **kwargs)

    def _sync_call(self, name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        loop = self._ensure_event_loop()
        res = loop.run_until_complete(self._call(name, args, kwargs))
        if inspect.isasyncgen(res):
            return async_gen_to_sync(res, loop=loop)
        return res

    async def _get_async_stream(
        self, endpoint: ClientEndpoint, *args: t.Any, **kwargs: t.Any
    ) -> t.AsyncGenerator[t.Any, None]:
        resp = await self._call(endpoint.name, args, kwargs)
        assert inspect.isasyncgen(resp)
        async for data in resp:
            yield data
