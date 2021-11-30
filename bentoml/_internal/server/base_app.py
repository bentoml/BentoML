import abc
import typing as t
import logging
from typing import TYPE_CHECKING

from starlette.responses import PlainTextResponse
from starlette.exceptions import HTTPException

if TYPE_CHECKING:  # pragma: no cover
    from starlette.routing import BaseRoute
    from starlette.responses import Response
    from starlette.middleware import Middleware
    from starlette.applications import Starlette

logger = logging.getLogger(__name__)


class BaseAppFactory(abc.ABC):
    name: str
    _is_ready: bool = False

    def mark_as_ready(self) -> None:
        self._is_ready = True

    async def livez(self, request) -> "Response":  # pylint: disable=unused-argument
        """
        Health check for BentoML API server.
        Make sure it works with Kubernetes liveness probe
        """
        return PlainTextResponse("\n", status_code=200)

    async def readyz(self, request) -> "Response":  # pylint: disable=unused-argument
        if self._is_ready:
            return PlainTextResponse("\n", status_code=200)
        raise HTTPException(500)

    @abc.abstractmethod
    def __call__(self) -> "Starlette":
        ...

    def routes(self) -> t.List["BaseRoute"]:
        from starlette.routing import Route

        routes = []
        routes.append(Route(path="/livez", endpoint=self.livez))
        routes.append(Route(path="/healthz", endpoint=self.livez))
        routes.append(Route(path="/readyz", endpoint=self.readyz))
        return routes

    def middlewares(self) -> t.List["Middleware"]:
        # return [InstrumentMiddleware()]  #TODO(jiang)
        return []
