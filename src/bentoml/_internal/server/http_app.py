from __future__ import annotations

import asyncio
import functools
import logging
import os
import sys
import typing as t

from simple_di import Provide
from simple_di import inject
from starlette.exceptions import HTTPException
from starlette.responses import PlainTextResponse

from ...exceptions import BentoMLException
from ..configuration.containers import BentoMLContainer
from ..context import trace_context
from ..server.base_app import BaseAppFactory
from ..service.service import Service

if t.TYPE_CHECKING:
    from opentelemetry.sdk.trace import Span
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import BaseRoute

    from ..service.inference_api import InferenceAPI

    LifecycleHook = t.Callable[[], None | t.Coroutine[t.Any, t.Any, None]]


logger = logging.getLogger(__name__)

DEFAULT_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>BentoML Prediction Service</title>
    <!-- Google Tag Manager -->
    <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
    new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
    j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
    'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
    })(window,document,'script','dataLayer','GTM-WNPGWRM');</script>
    <!-- End Google Tag Manager -->
    <link rel="stylesheet" type="text/css" href="./static_content/swagger-ui.css" />
    <link rel="stylesheet" type="text/css" href="./static_content/index.css" />
    <link rel="icon" type="image/png" href="./static_content/favicon-32x32.png" sizes="32x32" />
    <link rel="icon" type="image/png" href="./static_content/favicon-96x96.png" sizes="96x96" />
  </head>
  <body>
    <!-- Google Tag Manager (noscript) -->
    <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-WNPGWRM"
    height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
    <!-- End Google Tag Manager (noscript) -->
    <div id="swagger-ui"></div>
    <script src="./static_content/swagger-ui-bundle.js" charset="UTF-8"> </script>
    <script src="./static_content/swagger-ui-standalone-preset.js" charset="UTF-8"> </script>
    <script src="./static_content/swagger-initializer.js" charset="UTF-8"> </script>
    <div class="version">
        <div class="version-section"><a href="https://github.com/bentoml/BentoML" class="github-corner" aria-label="Powered by BentoML">
            <svg width="80" height="80" viewBox="0 0 250 250" style="fill:#151513; color:#fff; position: absolute; top: 0; border: 0; right: 0;" aria-hidden="true">
            <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
            <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path>
            <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path>
            </svg></a>
        </div>
    </div>
  </body>
</html>
"""


def log_exception(request: Request, exc_info: t.Any) -> None:
    """
    Logs an exception.  This is called by :meth:`handle_exception`
    if debugging is disabled and right before the handler is called.
    The default implementation logs the exception as error on the
    :attr:`logger`.
    """
    logger.error(
        "Exception on %s [%s]", request.url.path, request.method, exc_info=exc_info
    )


class HTTPAppFactory(BaseAppFactory):
    """
    HTTPApp creates a REST API server based on APIs defined with a BentoService
    via BentoService#apis. Each InferenceAPI will become one
    endpoint exposed on the REST server, and the RequestHandler defined on each
    InferenceAPI object will be used to handle Request object before feeding the
    request data into a Service API function
    """

    @inject
    def __init__(
        self,
        bento_service: Service,
        enable_access_control: bool = Provide[BentoMLContainer.http.cors.enabled],
        access_control_options: dict[str, list[str] | str | int] = Provide[
            BentoMLContainer.access_control_options
        ],
        enable_metrics: bool = Provide[
            BentoMLContainer.api_server_config.metrics.enabled
        ],
        timeout: int = Provide[BentoMLContainer.api_server_config.traffic.timeout],
        max_concurrency: int
        | None = Provide[BentoMLContainer.api_server_config.traffic.max_concurrency],
    ):
        self.bento_service = bento_service
        self.enable_access_control = enable_access_control
        self.access_control_options = access_control_options
        self.enable_metrics = enable_metrics
        super().__init__(timeout=timeout, max_concurrency=max_concurrency)

    @property
    def name(self) -> str:
        return self.bento_service.name

    async def index_view_func(self, _: Request) -> Response:
        """
        The default index view for BentoML API server. This includes the readme
        generated from docstring and swagger UI
        """
        from starlette.responses import Response

        # TODO: add readme description.
        return Response(
            content=DEFAULT_INDEX_HTML,
            status_code=200,
            media_type="text/html",
        )

    async def docs_view_func(self, _: Request) -> Response:
        from starlette.responses import JSONResponse

        return JSONResponse(self.bento_service.openapi_spec.asdict())

    @property
    def routes(self) -> list[BaseRoute]:
        """
        Setup routes for bento model server, including:

        /               Index Page, shows readme docs, metadata, and Swagger UI
        /docs.json      Returns Swagger/OpenAPI definition file in json format
        /healthz        liveness probe endpoint
        /readyz         Readiness probe endpoint
        /metrics        Prometheus metrics endpoint

        And user defined InferenceAPI list into routes, e.g.:
        /classify
        /predict
        """
        from starlette.routing import Mount
        from starlette.routing import Route
        from starlette.staticfiles import StaticFiles

        routes = super().routes

        routes.append(Route(path="/", name="home", endpoint=self.index_view_func))
        routes.append(
            Route(
                path="/docs.json",
                name="docs",
                endpoint=self.docs_view_func,
            )
        )

        parent_dir_path = os.path.dirname(os.path.realpath(__file__))
        routes.append(
            Mount(
                "/static_content",
                app=StaticFiles(
                    directory=os.path.join(parent_dir_path, "static_content")
                ),
                name="static_content",
            )
        )

        for _, api in self.bento_service.apis.items():
            api_route_endpoint = self._create_api_endpoint(api)
            if api.route.startswith("/"):
                route_path = api.route
            else:
                route_path = f"/{api.route}"

            routes.append(
                Route(
                    path=route_path,
                    name=api.name,
                    endpoint=api_route_endpoint,
                    methods=api.input.HTTP_METHODS,
                )
            )

        return routes

    @property
    def middlewares(self) -> list[Middleware]:
        middlewares = super().middlewares

        from starlette.middleware import Middleware

        for middleware_cls, options in self.bento_service.middlewares:
            middlewares.append(Middleware(middleware_cls, **options))

        if self.enable_access_control:
            assert (
                self.access_control_options.get("allow_origins") is not None
            ), "To enable cors, access_control_allow_origin must be set"

            from starlette.middleware.cors import CORSMiddleware

            middlewares.append(
                Middleware(CORSMiddleware, **self.access_control_options)
            )

        # metrics middleware
        if self.enable_metrics:
            from .http.instruments import HTTPTrafficMetricsMiddleware

            middlewares.append(Middleware(HTTPTrafficMetricsMiddleware))

        # otel middleware
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        def client_request_hook(span: Span, _: dict[str, t.Any]) -> None:
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

        access_log_config = BentoMLContainer.api_server_config.logging.access
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
    def on_startup(self) -> list[LifecycleHook]:
        on_startup = [
            *self.bento_service.startup_hooks,
            self.bento_service.on_asgi_app_startup,
        ]
        if BentoMLContainer.development_mode.get():
            for runner in self.bento_service.runners:
                on_startup.append(functools.partial(runner.init_local, quiet=True))
        else:
            for runner in self.bento_service.runners:
                if runner.embedded:
                    on_startup.append(functools.partial(runner.init_local, quiet=True))
                else:
                    on_startup.append(runner.init_client)
        on_startup.extend(super().on_startup)
        return on_startup

    async def readyz(self, _: "Request") -> "Response":
        if BentoMLContainer.api_server_config.runner_probe.enabled.get():
            runner_statuses = (
                runner.runner_handle_is_ready() for runner in self.bento_service.runners
            )
            runners_ready = all(await asyncio.gather(*runner_statuses))

            if not runners_ready:
                raise HTTPException(status_code=503, detail="Runners are not ready.")

        return PlainTextResponse("\n", status_code=200)

    @property
    def on_shutdown(self) -> list[LifecycleHook]:
        on_shutdown = [
            *self.bento_service.shutdown_hooks,
            self.bento_service.on_asgi_app_shutdown,
        ]
        for runner in self.bento_service.runners:
            on_shutdown.append(runner.destroy)
        on_shutdown.extend(super().on_shutdown)
        return on_shutdown

    def __call__(self) -> Starlette:
        app = super().__call__()
        for mount_app, path, name in self.bento_service.mount_apps:
            app.mount(app=mount_app, path=path, name=name)
        return app

    def _create_api_endpoint(
        self,
        api: InferenceAPI,
    ) -> t.Callable[[Request], t.Coroutine[t.Any, t.Any, Response]]:
        """
        Create api function for flask route, it wraps around user defined API
        callback and adapter class, and adds request logging and instrument metrics
        """
        from starlette.concurrency import run_in_threadpool  # type: ignore
        from starlette.responses import JSONResponse

        from ..utils import is_async_callable

        async def api_func(request: Request) -> Response:
            # handle_request may raise 4xx or 5xx exception.
            output = None
            with self.bento_service.context.in_request(request) as ctx:
                try:
                    input_data = await api.input.from_http_request(request)
                    if api.multi_input:
                        if api.needs_ctx:
                            input_data[api.ctx_param] = ctx
                        if is_async_callable(api.func):
                            output = await api.func(**input_data)
                        else:
                            output = await run_in_threadpool(api.func, **input_data)
                    else:
                        args = (input_data,)
                        if api.needs_ctx:
                            args = (input_data, ctx)
                        if is_async_callable(api.func):
                            output = await api.func(*args)
                        else:
                            output = await run_in_threadpool(api.func, *args)

                    response = await api.output.to_http_response(
                        output, ctx if api.needs_ctx else None
                    )

                    if trace_context.request_id is not None:
                        response.headers["X-BentoML-Request-ID"] = str(
                            trace_context.request_id
                        )
                    if (
                        BentoMLContainer.http.response.trace_id.get()
                        and trace_context.trace_id is not None
                    ):
                        response.headers["X-BentoML-Trace-ID"] = str(
                            trace_context.trace_id
                        )
                except BentoMLException as e:
                    log_exception(request, sys.exc_info())
                    if output is not None:
                        import inspect

                        signature = inspect.signature(api.output.to_proto)
                        param = next(iter(signature.parameters.values()))
                        ann = ""
                        if param is not inspect.Parameter.empty:
                            ann = param.annotation

                        # more descriptive errors if output is available
                        logger.error(
                            "Function '%s' has 'input=%s,output=%s' as IO descriptor, and returns 'result=%s', while expected return type is '%s'",
                            api.name,
                            api.input,
                            api.output,
                            type(output),
                            ann,
                        )

                    status = e.error_code.value
                    if 400 <= status < 500 and status not in (401, 403):
                        response = JSONResponse(
                            content="BentoService error handling API request: %s"
                            % str(e),
                            status_code=status,
                        )
                    else:
                        response = JSONResponse("", status_code=status)
                except asyncio.CancelledError:
                    # Special handling for Python 3.7 compatibility
                    raise
                except Exception:  # pylint: disable=broad-except
                    # For all unexpected error, return 500 by default. For example,
                    # if users' model raises an error of division by zero.
                    log_exception(request, sys.exc_info())

                    response = JSONResponse(
                        "An error has occurred in BentoML user code when handling this request, find the error details in server logs",
                        status_code=500,
                    )
                return response

        return api_func
