import logging

from simple_di import Provide, inject

from bentoml._internal.server.base_app import BaseApp

from ..configuration.containers import BentoMLContainer

feedback_logger = logging.getLogger("bentoml.feedback")
logger = logging.getLogger(__name__)


class RunnerApp(BaseApp):
    @inject
    def __init__(
        self,
        bundle_path: str = Provide[BentoMLContainer.bundle_path],
        runner_name: str = None,
        tracer=Provide[BentoMLContainer.tracer],
    ):
        from starlette.applications import Starlette

        from bentoml.saved_bundle.loader import load_from_dir

        assert bundle_path, repr(bundle_path)

        self.bento_service = load_from_dir(bundle_path)
        self.runner = self.bento_service.get_runner(runner_name)
        self.app_name = self.runner.name

        self.asgi_app = Starlette(
            on_startup=[self.on_startup], on_shutdown=[self.on_shutdown]
        )
        self.tracer = tracer

        self.setup_routes()

    def on_startup(self):
        pass

    def on_shutdown(self):
        pass

    async def run(self, request):
        ...

    async def run_batch(self, request):
        ...

    def setup_routes(self):
        self.asgi_app.add_route(path="/run", name="run", route=self.run)
        self.asgi_app.add_route(
            path="/run_batch", name="run_batch", route=self.run_batch
        )

    def get_app(self):

        return self.asgi_app
