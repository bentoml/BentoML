from __future__ import annotations

import functools
import inspect
import logging
import os
import sys
import typing as t

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

from bentoml._internal.server.http_app import log_exception

from ...exceptions import BentoMLException
from ..configuration.containers import BentoMLContainer
from ..context import trace_context
from .base_app import BaseAppFactory

if t.TYPE_CHECKING:
    from opentelemetry.sdk.trace import Span
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import BaseRoute

    from ..service import Service
    from ..service.inference_api import ServiceEndpoint
    from ..types import LifecycleHook


class UnifiedAppFactory(BaseAppFactory):
    def __init__(
        self,
        service: Service,
        *,
        timeout: int | None = None,
        max_concurrency: int | None = None,
        enable_metrics: bool = False,
    ) -> None:
        self.service = service
        self.enable_metrics = enable_metrics
        super().__init__(timeout=timeout, max_concurrency=max_concurrency)

    def __call__(self, is_main: bool = False) -> Starlette:
        app = super().__call__()
        if is_main:
            app.add_route("/schema.json", self.schema_view, name="schema")
            parent_dir_path = os.path.dirname(os.path.realpath(__file__))
            app.mount(
                "/static_content",
                app=StaticFiles(
                    directory=os.path.join(parent_dir_path, "static_content")
                ),
                name="static_content",
            )
        for mount_app, path, name in self.service.mount_apps:
            app.mount(app=mount_app, path=path, name=name)
        return app

    @property
    def name(self) -> str:
        return self.service.name

    @property
    def middlewares(self) -> list[Middleware]:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        middlewares = super().middlewares

        for middleware_cls, options in self.service.middlewares:
            middlewares.append(Middleware(middleware_cls, **options))

        def client_request_hook(span: Span | None, _scope: dict[str, t.Any]) -> None:
            from ..context import trace_context

            if span is not None:
                trace_context.request_id = span.context.span_id

        middlewares.append(
            Middleware(
                OpenTelemetryMiddleware,
                excluded_urls=BentoMLContainer.tracing_excluded_urls.get(),
                default_span_details=None,
                server_request_hook=None,
                client_request_hook=client_request_hook,
                tracer_provider=BentoMLContainer.tracer_provider.get(),
            )
        )

        if self.enable_metrics:
            from .http.instruments import RunnerTrafficMetricsMiddleware

            middlewares.append(Middleware(RunnerTrafficMetricsMiddleware))

        access_log_config = BentoMLContainer.runners_config.logging.access
        if access_log_config.enabled.get():
            from .http.access import AccessLogMiddleware

            access_logger = logging.getLogger("bentoml.access")
            if access_logger.getEffectiveLevel() <= logging.INFO:
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

    @property
    def on_shutdown(self) -> list[LifecycleHook]:
        on_shutdown = [
            *self.service.shutdown_hooks,
            self.service.on_asgi_app_shutdown,
        ]
        for runner in self.service.runners:
            on_shutdown.append(runner.destroy)
        on_shutdown.extend(super().on_shutdown)
        return on_shutdown

    @property
    def on_startup(self) -> list[LifecycleHook]:
        from ..ionext.runner_handle import RemoteRunnerHandle

        on_startup = [
            *self.service.startup_hooks,
            self.service.on_asgi_app_startup,
        ]
        for runner in self.service.runners:
            if runner.embedded:
                on_startup.append(functools.partial(runner.init_local, quiet=True))
            else:
                on_startup.append(
                    functools.partial(
                        runner.init_client, handle_class=RemoteRunnerHandle
                    )
                )
        on_startup.extend(super().on_startup)
        return on_startup

    async def schema_view(self, request: Request) -> Response:
        from starlette.responses import JSONResponse

        schema = self.service.schema
        return JSONResponse(schema)

    @property
    def routes(self) -> list[BaseRoute]:
        routes = super().routes

        for endpoint in self.service.endpoints.values():
            api_endpoint = functools.partial(self.api_endpoint, endpoint)
            route_path = endpoint.route
            if not route_path.startswith("/"):
                route_path = "/" + route_path
            routes.append(
                Route(route_path, api_endpoint, methods=["POST"], name=endpoint.name)
            )
        return routes

    async def api_endpoint(
        self, endpoint: ServiceEndpoint, request: Request
    ) -> Response:
        from starlette.concurrency import run_in_threadpool
        from starlette.responses import JSONResponse

        from ..ionext.serde import ALL_SERDE
        from ..utils import is_async_callable
        from ..utils.http import set_cookies

        media_type = request.headers.get("Content-Type", "application/json")
        serde = ALL_SERDE[media_type]()

        with self.service.context.in_request(request) as ctx:
            input_data = await endpoint.input_spec.from_http_request(request, serde)
            input_params = {k: getattr(input_data, k) for k in input_data.model_fields}
            if endpoint.ctx_param is not None:
                input_params[endpoint.ctx_param] = ctx

            try:
                if is_async_callable(endpoint.func):
                    output = await endpoint.func(**input_params)
                elif inspect.isasyncgenfunction(endpoint.func):
                    output = endpoint.func(**input_params)
                else:
                    output = await run_in_threadpool(endpoint.func, **input_params)

                response = await endpoint.output_spec.to_http_response(output, serde)
                response.headers.update(
                    {"Server": f"{self.service.service_str}/{endpoint.name}"}
                )

                if endpoint.ctx_param is not None:
                    response.status_code = ctx.response.status_code
                    response.headers.update(ctx.response.metadata)
                    set_cookies(response, ctx.response.cookies)
                if trace_context.request_id is not None:
                    response.headers["X-BentoML-Request-ID"] = str(
                        trace_context.request_id
                    )
                if (
                    BentoMLContainer.http.response.trace_id.get()
                    and trace_context.trace_id is not None
                ):
                    response.headers["X-BentoML-Trace-ID"] = str(trace_context.trace_id)
            except BentoMLException as e:
                log_exception(request, sys.exc_info())

                status = e.error_code.value
                if 400 <= status < 500 and status not in (401, 403):
                    response = JSONResponse(
                        content="BentoService error handling API request: %s" % str(e),
                        status_code=status,
                    )
                else:
                    response = JSONResponse("", status_code=status)
            except Exception:  # pylint: disable=broad-except
                # For all unexpected error, return 500 by default. For example,
                # if users' model raises an error of division by zero.
                log_exception(request, sys.exc_info())

                response = JSONResponse(
                    "An error has occurred in BentoML user code when handling this request, find the error details in server logs",
                    status_code=500,
                )
            return response
