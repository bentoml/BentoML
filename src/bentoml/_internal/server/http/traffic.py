from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from typing import Any

from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from ... import external_typing as ext


class TimeoutMiddleware:
    def __init__(self, app: ext.ASGIApp, timeout: float) -> None:
        self.app = app
        self.timeout = timeout

    def _set_timer_out(self, waiter: asyncio.Future[Any]) -> None:
        if not waiter.done():
            waiter.set_exception(asyncio.TimeoutError)

    async def __call__(
        self, scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        loop.call_later(self.timeout, self._set_timer_out, waiter)

        async def _send(message: ext.ASGIMessage) -> None:
            if not waiter.done():
                waiter.set_result(None)
            await send(message)

        fut = asyncio.ensure_future(self.app(scope, receive, _send), loop=loop)

        try:
            await waiter
        except asyncio.TimeoutError:
            if fut.cancel():
                resp = JSONResponse(
                    {
                        "error": f"Not able to process the request in {self.timeout} seconds"
                    },
                    status_code=504,
                )
                await resp(scope, receive, send)
        else:
            await fut  # wait for the future to finish


class MaxConcurrencyMiddleware:
    BYPASS_PATHS = frozenset({"/metrics", "/healthz", "/livez", "/readyz"})

    def __init__(self, app: ext.ASGIApp, max_concurrency: int) -> None:
        self.app = app
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def __call__(
        self, scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        if scope["path"] in self.BYPASS_PATHS:
            return await self.app(scope, receive, send)

        if self._semaphore.locked():
            resp = JSONResponse({"error": "Too many requests"}, status_code=429)
            await resp(scope, receive, send)
            return

        async with self._semaphore:
            await self.app(scope, receive, send)
