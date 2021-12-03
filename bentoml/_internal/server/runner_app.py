import json
import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml._internal.runner.utils import PAYLOAD_META_HEADER
from bentoml._internal.runner.utils import multipart_to_payload_params
from bentoml._internal.runner.container import AutoContainer

from ..server.base_app import BaseAppFactory
from ..configuration.containers import BentoMLContainer

feedback_logger = logging.getLogger("bentoml.feedback")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from starlette.routing import BaseRoute
    from starlette.requests import Request
    from starlette.responses import Response

    from ..runner import Runner
    from ..tracing import Tracer


class RunnerAppFactory(BaseAppFactory):
    @inject
    def __init__(
        self,
        runner: "Runner",
        instance_id: t.Optional[int] = None,
        tracer: "Tracer" = Provide[BentoMLContainer.tracer],
    ) -> None:
        self.runner = runner
        self.instance_id = instance_id
        self.tracer = tracer

    @property
    def name(self) -> str:
        return self.runner.name

    @property
    def on_startup(self) -> t.List[t.Callable[[], None]]:
        on_startup = super().on_startup
        on_startup.insert(0, self.runner._impl.setup)  # type: ignore[reportPrivateUsage]
        return on_startup

    @property
    def routes(self) -> t.List["BaseRoute"]:
        """
        Setup routes for Runner server, including:

        /healthz        liveness probe endpoint
        /readyz         Readiness probe endpoint
        /metrics        Prometheus metrics endpoint

        /run
        /run_batch
        """
        from starlette.routing import Route

        routes = super().routes
        routes.append(Route("/run", self.async_run, methods=["POST"]))
        routes.append(Route("/run_batch", self.async_run_batch, methods=["POST"]))
        return routes

    async def async_run(self, request: "Request") -> "Response":
        from starlette.responses import Response

        assert self._is_ready

        params = await multipart_to_payload_params(request)
        params = params.map(AutoContainer.payload_to_single)
        ret = await self.runner.async_run(*params.args, **params.kwargs)
        payload = AutoContainer.single_to_payload(ret)
        return Response(
            payload.data,
            headers={
                PAYLOAD_META_HEADER: json.dumps(payload.meta),
                "Server": f"BentoML-Runner/{self.runner.name}/{self.instance_id}",
            },
        )

    async def async_run_batch(self, request: "Request") -> "Response":
        from starlette.responses import Response

        assert self._is_ready

        params = await multipart_to_payload_params(request)
        params = params.map(AutoContainer.payload_to_batch)
        ret = await self.runner.async_run_batch(*params.args, **params.kwargs)
        payload = AutoContainer.batch_to_payload(ret)
        return Response(
            payload.data,
            headers={
                PAYLOAD_META_HEADER: json.dumps(payload.meta),
                "Server": f"BentoML-Runner/{self.runner.name}/{self.instance_id}",
            },
        )
