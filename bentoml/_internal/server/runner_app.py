from __future__ import annotations

import json
import typing as t
import asyncio
import logging
import functools
from typing import TYPE_CHECKING
from functools import partial

from ..trace import ServiceContext
from ..runner.utils import PAYLOAD_META_HEADER
from ..runner.utils import multipart_to_payload_params
from ..runner.utils import payload_paramss_to_batch_params
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

    from ..runner.runner import Runner
    from ..runner.runner import RunnerMethod


class RunnerAppFactory(BaseAppFactory):
    def __init__(
        self,
        runner: Runner,
        worker_index: int = 0,
    ) -> None:
        self.runner = runner
        self.worker_index = worker_index

        from starlette.responses import Response

        TooManyRequests = partial(Response, status_code=429)

        self.dispatchers: dict[str, CorkDispatcher] = {}
        for method in runner.runner_methods:
            if not method.config.batchable:
                continue
            self.dispatchers[method.name] = CorkDispatcher(
                max_latency_in_ms=method.max_latency_ms,
                max_batch_size=method.max_batch_size,
                fallback=TooManyRequests,
            )

    @property
    def name(self) -> str:
        return self.runner.name

    @property
    def on_startup(self) -> t.List[t.Callable[[], None]]:
        on_startup = super().on_startup
        on_startup.insert(0, functools.partial(self.runner.init_local, quiet=True))
        on_startup.insert(
            0,
            functools.partial(
                self.runner.scheduling_strategy.setup_worker,
                runnable_class=self.runner.runnable_class,
                resource_request=self.runner.get_effective_resource_config(),
                worker_index=self.worker_index,
            ),
        )
        return on_startup

    @property
    def on_shutdown(self) -> t.List[t.Callable[[], None]]:
        on_shutdown = [self.runner.destroy]
        for dispatcher in self.dispatchers.values():
            on_shutdown.append(dispatcher.shutdown)
        on_shutdown.extend(super().on_shutdown)
        return on_shutdown

    @property
    def routes(self) -> t.List[BaseRoute]:
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
        for method in self.runner.runner_methods:
            path = "/" if method.name == "__call__" else "/" + method.name
            if method.config.batchable:
                _func = self.dispatchers[method.name](
                    self._async_cork_run(runner_method=method)
                )
                routes.append(
                    Route(
                        path=path,
                        endpoint=_func,
                        methods=["POST"],
                    )
                )
            else:
                routes.append(
                    Route(
                        path=path,
                        endpoint=self.async_run(runner_method=method),
                        methods=["POST"],
                    )
                )
        return routes

    @property
    def middlewares(self) -> list[Middleware]:
        middlewares = super().middlewares

        # otel middleware
        import opentelemetry.instrumentation.asgi as otel_asgi  # type: ignore[import]
        from starlette.middleware import Middleware

        def client_request_hook(span: Span, _scope: t.Dict[str, t.Any]) -> None:
            if span is not None:
                span_id: int = span.context.span_id
                ServiceContext.request_id_var.set(span_id)

        def client_response_hook(span: Span, _message: t.Any) -> None:
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

    def _async_cork_run(
        self,
        runner_method: RunnerMethod[t.Any, t.Any, t.Any],
    ) -> t.Callable[[t.Iterable[Request]], t.Coroutine[None, None, list[Response]]]:
        from starlette.responses import Response

        async def _run(requests: t.Iterable[Request]) -> list[Response]:
            assert self._is_ready
            if not requests:
                return []
            params_list = await asyncio.gather(
                *tuple(multipart_to_payload_params(r) for r in requests)
            )

            input_batch_dim, output_batch_dim = runner_method.config.batch_dim

            batched_params, indices = payload_paramss_to_batch_params(
                params_list,
                input_batch_dim,
            )

            batch_ret = await runner_method.async_run(
                *batched_params.args,
                **batched_params.kwargs,
            )

            payloads = AutoContainer.batch_to_payloads(
                batch_ret,
                indices,
                batch_dim=output_batch_dim,
            )

            return [
                Response(
                    payload.data,
                    headers={
                        PAYLOAD_META_HEADER: json.dumps(payload.meta),
                        "Content-Type": f"application/vnd.bentoml.{payload.container}",
                        "Server": f"BentoML-Runner/{self.runner.name}/{runner_method.name}/{self.worker_index}",
                    },
                )
                for payload in payloads
            ]

        return _run

    def async_run(
        self,
        runner_method: RunnerMethod[t.Any, t.Any, t.Any],
    ) -> t.Callable[[Request], t.Coroutine[None, None, Response]]:
        from starlette.responses import Response

        async def _run(request: Request) -> Response:
            assert self._is_ready

            logger.info(request)
            params = await multipart_to_payload_params(request)
            params = params.map(AutoContainer.from_payload)
            ret = await runner_method.async_run(*params.args, **params.kwargs)

            payload = AutoContainer.to_payload(ret, 0)
            return Response(
                payload.data,
                headers={
                    PAYLOAD_META_HEADER: json.dumps(payload.meta),
                    "Content-Type": f"application/vnd.bentoml.{payload.container}",
                    "Server": f"BentoML-Runner/{self.runner.name}/{runner_method.name}/{self.worker_index}",
                },
            )

        return _run
