import asyncio
import logging
import os
import sys
import typing as t

from simple_di import Provide, inject

from bentoml._internal.server.base_app import BaseAppFactory
from bentoml._internal.service.service import Service
from bentoml.exceptions import BentoMLException

from ..configuration.containers import BentoMLContainer, BentoServerContainer

if t.TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import BaseRoute

    from bentoml._internal.server.metrics.prometheus import PrometheusClient
    from bentoml._internal.service.inference_api import InferenceAPI
    from bentoml._internal.tracing import Tracer


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
    via BentoService#get_service_apis call. Each InferenceAPI will become one
    endpoint exposed on the REST server, and the RequestHandler defined on each
    InferenceAPI object will be used to handle Request object before feeding the
    request data into a Service API function
    """

    @inject
    def __init__(
        self,
        bento_service: Service,
        enable_metrics: bool = Provide[BentoServerContainer.config.metrics.enabled],
        tracer: "Tracer" = Provide[BentoMLContainer.tracer],
        metrics_client: "PrometheusClient" = Provide[
            BentoServerContainer.metrics_client
        ],
        enable_access_control: bool = Provide[BentoServerContainer.config.cors.enabled],
        access_control_options: t.Dict = Provide[
            BentoServerContainer.access_control_options
        ],
    ) -> None:
        self.bento_service = bento_service

        self.app_name = bento_service.name
        self.enable_metrics = enable_metrics
        self.tracer = tracer
        self.metrics_client = metrics_client
        self.enable_access_control = enable_access_control
        self.access_control_options = access_control_options

    async def index_view_func(
        self, request
    ) -> "Response":  # pylint: disable=unused-argument
        """
        The default index view for BentoML API server. This includes the readme
        generated from docstring and swagger UI
        """
        from starlette.responses import Response

        return Response(
            content=DEFAULT_INDEX_HTML.format(readme=self.bento_service.__doc__),
            status_code=200,
            media_type="text/html",
        )

    @inject
    async def metrics_view_func(
        self, request
    ) -> "Response":  # pylint: disable=unused-argument
        from starlette.responses import Response

        return Response(
            self.metrics_client.generate_latest(),
            media_type=self.metrics_client.CONTENT_TYPE_LATEST,
        )

    async def docs_view_func(
        self, request
    ) -> "Response":  # pylint: disable=unused-argument
        from starlette.responses import JSONResponse

        docs = self.bento_service.openapi_doc()
        return JSONResponse(docs)

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
        from starlette.routing import Mount, Route
        from starlette.staticfiles import StaticFiles

        routes = super().routes()
        routes.append(Route(path="/", name="home", endpoint=self.index_view_func))
        routes.append(
            Route(
                path="/docs.json",
                name="docs",
                endpoint=self.docs_view_func,
            )
        )

        if self.enable_metrics:
            routes.append(
                Route(path="/metrics", name="metrics", endpoint=self.metrics_view_func)
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

        for _, api in self.bento_service._apis.items():
            api_route_endpoint = self.create_api_endpoint(api)
            routes.append(
                Route(
                    path="/{}".format(api.route),
                    name=api.name,
                    endpoint=api_route_endpoint,
                    methods=api.input.HTTP_METHODS,
                )
            )

        return routes

    def middlewares(self) -> t.List["Middleware"]:
        from starlette.middleware import Middleware

        middlewares = super().middlewares()

        for middleware_cls, options in self.bento_service._middlewares:
            middlewares.append(Middleware(middleware_cls, **options))

        if self.enable_access_control:
            assert (
                self.access_control_options.get("allow_origins") is not None
            ), "To enable cors, access_control_allow_origin must be set"

            from starlette.middleware.cors import CORSMiddleware

            middlewares.append(
                Middleware(CORSMiddleware, **self.access_control_options)
            )

        return middlewares

    def __call__(self) -> "Starlette":
        from starlette.applications import Starlette

        app = Starlette(
            debug=True,  # TDOO: inject this from `debug=True` or `--production` flag
            routes=self.routes(),
            middleware=self.middlewares(),
            on_shutdown=[self.bento_service._on_asgi_app_shutdown],
            on_startup=[self.bento_service._on_asgi_app_startup, self.mark_as_ready],
        )

        for mount_app, path, name in self.bento_service._mount_apps:
            app.mount(app=mount_app, path=path, name=name)

        return app

    def create_api_endpoint(
        self, api: "InferenceAPI"
    ) -> t.Callable[["Request"], t.Coroutine[t.Any, t.Any, "Response"]]:
        """
        Create api function for flask route, it wraps around user defined API
        callback and adapter class, and adds request logging and instrument metrics
        """
        from starlette.concurrency import run_in_threadpool
        from starlette.responses import JSONResponse

        async def api_func(
            request: "Request",
        ) -> "Response":
            # handle_request may raise 4xx or 5xx exception.
            try:
                input_data = await api.input.from_http_request(request)
                if asyncio.iscoroutinefunction(api.func):
                    output = await api.func(input_data)
                else:
                    output = await run_in_threadpool(api.func, input_data)
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

        """
        TODO: instrument tracing
        def api_func_with_tracing():
            with self.tracer.span(
                service_name=f"BentoService.{self.bento_service.name}",
                span_name=f"InferenceAPI {api.name} HTTP route",
                request_headers=request.headers,
            ):
                return api_func()

        return api_func_with_tracing
        """
        return api_func
