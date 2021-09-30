import abc
import logging
from typing import TYPE_CHECKING

from starlette.exceptions import HTTPException
from starlette.responses import PlainTextResponse

if TYPE_CHECKING:
    from starlette.applications import Starlette

logger = logging.getLogger(__name__)


class BaseApp(abc.ABC):
    asgi_app: "Starlette"
    name: str
    _is_ready: bool = False

    @abc.abstractmethod
    def setup(self) -> None:
        ...

    def _setup(self) -> None:
        self.setup()
        self._is_ready = True

    @abc.abstractmethod
    def on_startup(self):
        pass

    @abc.abstractmethod
    def on_shutdown(self):
        pass

    async def livez(self):
        """
        Health check for BentoML API server.
        Make sure it works with Kubernetes liveness probe
        """
        return PlainTextResponse("\n", status_code=200)

    async def readyz(self):
        if self._is_ready:
            return PlainTextResponse("\n", status_code=200)
        raise HTTPException(500)

    @abc.abstractmethod
    def app_hook(self):
        self.asgi_app.add_route("/livez", self.livez)
        self.asgi_app.add_route("/healthz", self.livez)
        self.asgi_app.add_route("/readyz", self.readyz)
