import abc
import logging
from typing import TYPE_CHECKING

from starlette.exceptions import HTTPException
from starlette.responses import PlainTextResponse

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.responses import Response

logger = logging.getLogger(__name__)


class BaseApp(abc.ABC):
    app: "Starlette"
    name: str
    _is_ready: bool = False

    @abc.abstractmethod
    def setup(self) -> None:
        ...

    def on_startup(self) -> None:
        pass

    def on_shutdown(self) -> None:
        pass

    def setup(self) -> None:
        if self._is_ready:
            return

        self._setup()
        self._is_ready = True

    async def livez(self) -> "Response":
        """
        Health check for BentoML API server.
        Make sure it works with Kubernetes liveness probe
        """
        return PlainTextResponse("\n", status_code=200)

    async def readyz(self) -> "Response":
        if self._is_ready:
            return PlainTextResponse("\n", status_code=200)
        raise HTTPException(500)

    def setup_routes(self) -> None:
        self.app.add_route("/livez", self.livez)
        self.app.add_route("/healthz", self.livez)
        self.app.add_route("/readyz", self.readyz)
