import abc
import logging
import typing as t
from typing import TYPE_CHECKING

from starlette.exceptions import HTTPException
from starlette.responses import PlainTextResponse

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import Response
    from starlette.routing import BaseRoute

logger = logging.getLogger(__name__)


class BaseAppFactory(abc.ABC):
    name: str

    # TODO: set _is_ready state after initial setup call
    _is_ready: bool = False

    @abc.abstractmethod
    def setup(self) -> None:
        ...

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
