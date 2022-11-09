import abc
import typing as t
import logging
from typing import TYPE_CHECKING

from starlette.responses import PlainTextResponse
from starlette.exceptions import HTTPException

if TYPE_CHECKING:
    from starlette.routing import BaseRoute
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.middleware import Middleware
    from starlette.applications import Starlette


logger = logging.getLogger(__name__)


class BaseAppFactory(abc.ABC):
    _is_ready: bool = False

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    def on_startup(self) -> t.List[t.Callable[[], None]]:
        return [self.mark_as_ready]

    @property
    def on_shutdown(self) -> t.List[t.Callable[[], None]]:
        return []

    def mark_as_ready(self) -> None:
        self._is_ready = True

    async def livez(self, _: "Request") -> "Response":
        """
        Health check for BentoML API server.
        Make sure it works with Kubernetes liveness probe
        """
        return PlainTextResponse("\n", status_code=200)

    async def readyz(self, _: "Request") -> "Response":
        if self._is_ready:
            return PlainTextResponse("\n", status_code=200)
        raise HTTPException(500)

    def __call__(self) -> "Starlette":
        from starlette.applications import Starlette

        from ..configuration import get_debug_mode

        return Starlette(
            debug=get_debug_mode(),
            routes=self.routes,
            middleware=self.middlewares,
            on_startup=self.on_startup,
            on_shutdown=self.on_shutdown,
        )

    @property
    def routes(self) -> t.List["BaseRoute"]:
        from starlette.routing import Route

        routes: t.List["BaseRoute"] = []
        routes.append(Route(path="/livez", name="livez", endpoint=self.livez))
        routes.append(Route(path="/healthz", name="healthz", endpoint=self.livez))
        routes.append(Route(path="/readyz", name="readyz", endpoint=self.readyz))
        return routes

    @property
    def middlewares(self) -> t.List["Middleware"]:
        return []
