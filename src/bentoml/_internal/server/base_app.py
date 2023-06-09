from __future__ import annotations

import abc
import contextlib
import logging
import typing as t
from typing import TYPE_CHECKING

from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse

from ..utils import is_async_callable

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import BaseRoute

    from ..types import LifecycleHook


logger = logging.getLogger(__name__)


class BaseAppFactory(abc.ABC):
    _is_ready: bool = False

    def __init__(
        self, *, timeout: int | None = None, max_concurrency: int | None = None
    ) -> None:
        self.timeout = timeout
        self.max_concurrency = max_concurrency

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    def on_startup(self) -> list[LifecycleHook]:
        return [self.mark_as_ready]

    @property
    def on_shutdown(self) -> list[LifecycleHook]:
        return []

    def mark_as_ready(self) -> None:
        self._is_ready = True

    async def livez(self, _: Request) -> Response:
        """
        Health check for BentoML API server.
        Make sure it works with Kubernetes liveness probe
        """
        return PlainTextResponse("\n", status_code=200)

    async def readyz(self, _: Request) -> Response:
        if self._is_ready:
            return PlainTextResponse("\n", status_code=200)
        raise HTTPException(500)

    def __call__(self) -> Starlette:
        from starlette.applications import Starlette

        from ..configuration import get_debug_mode

        @contextlib.asynccontextmanager
        async def lifespan(_: Starlette) -> t.AsyncGenerator[None, None]:
            for on_startup in self.on_startup:
                if is_async_callable(on_startup):
                    await on_startup()
                else:
                    on_startup()
            yield
            for on_shutdown in self.on_shutdown:
                if is_async_callable(on_shutdown):
                    await on_shutdown()
                else:
                    on_shutdown()

        return Starlette(
            debug=get_debug_mode(),
            routes=self.routes,
            middleware=self.middlewares,
            lifespan=lifespan,
        )

    @property
    def routes(self) -> list[BaseRoute]:
        from starlette.routing import Route

        routes: list[BaseRoute] = []
        routes.append(Route(path="/livez", name="livez", endpoint=self.livez))
        routes.append(Route(path="/healthz", name="healthz", endpoint=self.livez))
        routes.append(Route(path="/readyz", name="readyz", endpoint=self.readyz))
        return routes

    @property
    def middlewares(self) -> list[Middleware]:
        from .http.traffic import MaxConcurrencyMiddleware
        from .http.traffic import TimeoutMiddleware

        results: list[Middleware] = []
        if self.timeout:
            results.append(Middleware(TimeoutMiddleware, timeout=self.timeout))
        if self.max_concurrency:
            results.append(
                Middleware(
                    MaxConcurrencyMiddleware, max_concurrency=self.max_concurrency
                )
            )
        return results
