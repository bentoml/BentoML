import os
import sys
import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ..context import trace_context
from ...exceptions import BentoMLException
from ..server.base_app import BaseAppFactory
from ..service.service import Service
from ..configuration.containers import DeploymentContainer
from ..io_descriptors.multipart import Multipart

if TYPE_CHECKING:
    from starlette.routing import BaseRoute
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.middleware import Middleware
    from starlette.applications import Starlette
    from opentelemetry.sdk.trace import Span

    from ..service.inference_api import InferenceAPI


feedback_logger = logging.getLogger("bentoml.feedback")
logger = logging.getLogger(__name__)

DEFAULT_INDEX_HTML = """\
<!DOCTYPE html>
<head>
  <link rel="stylesheet" type="text/css" href="static_content/main.css">
  <link rel="stylesheet" type="text/css" href="static_content/readme.css">
  <link rel="stylesheet" type="text/css" href="static_content/swagger-ui.css">
</head>
<body>
  <div id="tab">
    <button
      class="tabLinks active"
      onclick="openTab(event, 'swagger_ui_container')"
      id="defaultOpen"
    >
      Swagger UI
    </button>
    <button class="tabLinks" onclick="openTab(event, 'markdown_readme')">
      ReadMe
    </button>
  </div>
  <script>
    function openTab(evt, tabName) {{
      // Declare all variables
      var i, tabContent, tabLinks;
      // Get all elements with class="tabContent" and hide them
      tabContent = document.getElementsByClassName("tabContent");
      for (i = 0; i < tabContent.length; i++) {{
        tabContent[i].style.display = "none";
      }}

      // Get all elements with class="tabLinks" and remove the class "active"
      tabLinks = document.getElementsByClassName("tabLinks");
      for (i = 0; i < tabLinks.length; i++) {{
        tabLinks[i].className = tabLinks[i].className.replace(" active", "");
      }}

      // Show the current tab, and add an "active" class to the button that opened the
      // tab
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }}
  </script>
  <div id="markdown_readme" class="tabContent"></div>
  <script src="static_content/marked.min.js"></script>
  <script>
    // TODO: Fix readme escape and index page redesign
    var markdownContent = marked(`{readme}`);
    var element = document.getElementById('markdown_readme');
    element.innerHTML = markdownContent;
  </script>
  <div id="swagger_ui_container" class="tabContent" style="display: block"></div>
  <script src="static_content/swagger-ui-bundle.js"></script>
  <script>
      SwaggerUIBundle({{
          url: '/docs.json',
          dom_id: '#swagger_ui_container'
      }})
  </script>
</body>
"""


def log_exception(request: "Request", exc_info: t.Any) -> None:
    """
    Logs an exception.  This is called by :meth:`handle_exception`
    if debugging is disabled and right before the handler is called.
    The default implementation logs the exception as error on the
    :attr:`logger`.
    """
    logger.error(
        "Exception on %s [%s]", request.url.path, request.method, exc_info=exc_info
    )


class ServiceAppFactory(BaseAppFactory):
    """
    ServiceApp creates a REST API server based on APIs defined with a BentoService
    via BentoService#apis. Each InferenceAPI will become one
    endpoint exposed on the REST server, and the RequestHandler defined on each
    InferenceAPI object will be used to handle Request object before feeding the
    request data into a Service API function
    """

    @inject
    def __init__(
        self,
        bento_service: Service,
        enable_access_control: bool = Provide[
            DeploymentContainer.api_server_config.cors.enabled
        ],
        access_control_options: t.Dict[str, t.Union[t.List[str], int]] = Provide[
            DeploymentContainer.access_control_options
        ],
        enable_metrics: bool = Provide[
            DeploymentContainer.api_server_config.metrics.enabled
        ],
    ) -> None:
        self.bento_service = bento_service
        self.enable_access_control = enable_access_control
        self.access_control_options = access_control_options
        self.enable_metrics = enable_metrics

    @property
    def name(self) -> str:
        return self.bento_service.name

    async def index_view_func(self, _: "Request") -> "Response":
        """
        The default index view for BentoML API server. This includes the readme
        generated from docstring and swagger UI
        """
        from starlette.responses import Response

        return Response(
            content=DEFAULT_INDEX_HTML.format(readme=self.bento_service.doc),
            status_code=200,
            media_type="text/html",
        )

    async def docs_view_func(self, _: "Request") -> "Response":
        from starlette.responses import JSONResponse

        docs = self.bento_service.openapi_doc()
        return JSONResponse(docs)

    @property
    def routes(self) -> t.List["BaseRoute"]:
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
            routes.append(
                Route(
                    path="/{}".format(api.route),
                    name=api.name,
                    endpoint=api_route_endpoint,
                    methods=api.input.HTTP_METHODS,
                )
            )

        return routes

    @property
    def middlewares(self) -> t.List["Middleware"]:
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
            from .instruments import MetricsMiddleware

            middlewares.append(
                Middleware(
                    MetricsMiddleware,
                    bento_service=self.bento_service,
                )
            )

        # otel middleware
        import opentelemetry.instrumentation.asgi as otel_asgi  # type: ignore

        def client_request_hook(span: "Span", _scope: t.Dict[str, t.Any]) -> None:
            if span is not None:
                trace_context.request_id = span.context.span_id  # type: ignore

        def client_response_hook(span: "Span", _message: t.Any) -> None:
            if span is not None:
                del trace_context.request_id

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

        access_log_config = DeploymentContainer.api_server_config.logging.access
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

    @property
    def on_startup(self) -> t.List[t.Callable[[], None]]:
        on_startup = super().on_startup
        on_startup.insert(0, self.bento_service.on_asgi_app_startup)
        return on_startup

    @property
    def on_shutdown(self) -> t.List[t.Callable[[], None]]:
        on_shutdown = super().on_shutdown
        on_shutdown.insert(0, self.bento_service.on_asgi_app_shutdown)
        return on_shutdown

    def __call__(self) -> "Starlette":
        app = super().__call__()
        for mount_app, path, name in self.bento_service.mount_apps:
            app.mount(app=mount_app, path=path, name=name)
        return app

    @staticmethod
    def _create_api_endpoint(
        api: "InferenceAPI",
    ) -> t.Callable[["Request"], t.Coroutine[t.Any, t.Any, "Response"]]:
        """
        Create api function for flask route, it wraps around user defined API
        callback and adapter class, and adds request logging and instrument metrics
        """
        from starlette.responses import JSONResponse
        from starlette.concurrency import run_in_threadpool

        async def api_func(
            request: "Request",
        ) -> "Response":
            # handle_request may raise 4xx or 5xx exception.
            try:
                input_data = await api.input.from_http_request(request)
                if asyncio.iscoroutinefunction(api.func):
                    if isinstance(api.input, Multipart):
                        output = await api.func(**input_data)
                    else:
                        output = await api.func(input_data)
                else:
                    if isinstance(api.input, Multipart):
                        output: t.Any = await run_in_threadpool(api.func, **input_data)
                    else:
                        output: t.Any = await run_in_threadpool(api.func, input_data)
                response = await api.output.to_http_response(output)
            except BentoMLException as e:
                log_exception(request, sys.exc_info())

                if 400 <= e.error_code < 500 and e.error_code not in (401, 403):
                    response = JSONResponse(
                        content="BentoService error handling API request: %s" % str(e),
                        status_code=e.error_code,
                    )
                else:
                    response = JSONResponse("", status_code=e.error_code)
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
