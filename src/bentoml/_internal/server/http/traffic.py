from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from ... import external_typing as ext


class TimeoutMiddleware:
    def __init__(self, app: ext.ASGIApp, timeout: float) -> None:
        self.app = app
        self.timeout = timeout

    async def __call__(
        self, scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        try:
            await asyncio.wait_for(self.app(scope, receive, send), timeout=self.timeout)
        except asyncio.TimeoutError:
            resp = JSONResponse(
                {"error": f"Not able to process the request in {self.timeout} seconds"},
                status_code=504,
            )
            await resp(scope, receive, send)


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

        if self._semaphore.locked() and scope["path"] not in self.BYPASS_PATHS:
            resp = JSONResponse({"error": "Too many requests"}, status_code=429)
            await resp(scope, receive, send)
            return

        async with self._semaphore:
            await self.app(scope, receive, send)
