import logging
from typing import TYPE_CHECKING

from simple_di import Provide, inject

from bentoml._internal.server.base_app import BaseApp

from ..configuration.containers import BentoMLContainer

feedback_logger = logging.getLogger("bentoml.feedback")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response

    from bentoml._internal.runner import Runner


class RunnerApp(BaseApp):
    @inject
    def __init__(
        self,
        runner: "Runner",
        tracer=Provide[BentoMLContainer.tracer],
    ):
        self.runner = runner
        self.tracer = tracer

    @property
    def asgi_app(self) -> "Starlette":
        from starlette.applications import Starlette
        from starlette.routing import Route

        routes = [
            Route("/run", self.run, methods=["POST"]),
            Route("/run_batch", self.run_batch, methods=["POST"]),
        ]
        return Starlette(
            routes=routes, on_startup=[self.on_startup], on_shutdown=[self.on_shutdown]
        )

    @property
    def name(self) -> str:
        return self.runner.name

    def on_startup(self):
        pass

    def on_shutdown(self):
        pass

    async def run(self, request: "Request") -> "Response":
        form = await request.form()
        # TODO(jiang)

    async def run_batch(self, request: "Request") -> "Response":
        ...
