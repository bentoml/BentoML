import json
import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING
from functools import partial

from ..trace import ServiceContext
from ..runner.utils import Params
from ..runner.utils import PAYLOAD_META_HEADER
from ..runner.utils import multipart_to_payload_params
from ..server.base_app import BaseAppFactory
from ..runner.container import AutoContainer
from ..marshal.dispatcher import CorkDispatcher
from ..configuration.containers import DeploymentContainer

feedback_logger = logging.getLogger("bentoml.feedback")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from starlette.routing import BaseRoute
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.middleware import Middleware
    from opentelemetry.sdk.trace import Span

    from ..runner import Runner
    from ..runner import SimpleRunner


class RunnerAppFactory(BaseAppFactory):
    def __init__(
        self,
        runner: "t.Union[Runner, SimpleRunner]",
        instance_id: t.Optional[int] = None,
    ) -> None:
        self.runner = runner
        self.instance_id = instance_id

        from starlette.responses import Response

        from ..runner import Runner

        TooManyRequests = partial(Response, status_code=429)

        options = self.runner.batch_options
        if isinstance(self.runner, Runner) and options.enabled:
            options = self.runner.batch_options
            self.dispatcher = CorkDispatcher(
                max_latency_in_ms=options.max_latency_ms,
                max_batch_size=options.max_batch_size,
                fallback=TooManyRequests,
            )
        else:
            self.dispatcher = None

    @property
    def name(self) -> str:
        return self.runner.name

    @property
    def on_startup(self) -> t.List[t.Callable[[], None]]:
        on_startup = super().on_startup
        on_startup.insert(0, self.runner._impl.setup)  # type: ignore[reportPrivateUsage]
        return on_startup

    @property
    def on_shutdown(self) -> t.List[t.Callable[[], None]]:
        on_shutdown = super().on_shutdown
        on_shutdown.insert(0, self.runner._impl.shutdown)  # type: ignore[reportPrivateUsage]
        if self.dispatcher is not None:
            on_shutdown.insert(0, self.dispatcher.shutdown)
        return on_shutdown

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
        routes.append(Route("/run_batch", self.async_run_batch, methods=["POST"]))

        if self.dispatcher is not None:
            _func = self.dispatcher(self._async_cork_run)
            routes.append(Route("/run", _func, methods=["POST"]))
        else:
            routes.append(Route("/run", self.async_run, methods=["POST"]))
        return routes

    @property
    def middlewares(self) -> t.List["Middleware"]:
        middlewares = super().middlewares

        # otel middleware
        import opentelemetry.instrumentation.asgi as otel_asgi  # type: ignore[import]
        from starlette.middleware import Middleware

        def client_request_hook(span: "Span", _scope: t.Dict[str, t.Any]) -> None:
            if span is not None:
                span_id: int = span.context.span_id
                ServiceContext.request_id_var.set(span_id)

        def client_response_hook(span: "Span", _message: t.Any) -> None:
            if span is not None:
                ServiceContext.request_id_var.set(None)

        middlewares.append(
            Middleware(
                otel_asgi.OpenTelemetryMiddleware,
                excluded_urls=None,
                default_span_details=None,
                server_request_hook=None,
                client_request_hook=client_request_hook,
                client_response_hook=client_response_hook,
                tracer_provider=DeploymentContainer.tracer_provider.get(),
            )
        )

        access_log_config = DeploymentContainer.runners_config.logging.access
        if access_log_config.enabled.get():
            from .access import AccessLogMiddleware

            middlewares.append(
                Middleware(
                    AccessLogMiddleware,
                    has_request_content_length=access_log_config.request_content_length.get(),
                    has_request_content_type=access_log_config.request_content_type.get(),
                    has_response_content_length=access_log_config.response_content_length.get(),
                    has_response_content_type=access_log_config.response_content_type.get(),
                )
            )

        return middlewares

    async def _async_cork_run(
        self, requests: t.Iterable["Request"]
    ) -> t.List["Response"]:
        from starlette.responses import Response

        assert self._is_ready

        params_list = await asyncio.gather(
            *tuple(multipart_to_payload_params(r) for r in requests)
        )
        params = Params.agg(
            params_list,
            lambda i: AutoContainer.payloads_to_batch(
                i,
                batch_axis=self.runner.batch_options.input_batch_axis,
            ),
        )
        batch_ret = await self.runner.async_run_batch(*params.args, **params.kwargs)
        payloads = AutoContainer.batch_to_payloads(
            batch_ret,
            batch_axis=self.runner.batch_options.input_batch_axis,
        )
        return [
            Response(
                payload.data,
                headers={
                    PAYLOAD_META_HEADER: json.dumps(payload.meta),
                    "Server": f"BentoML-Runner/{self.runner.name}/{self.instance_id}",
                },
            )
            for payload in payloads
        ]

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
