from starlette.types import ASGIApp
from bentoml._internal.service.service import Service
import os
import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Dict
import typing as t

from google.protobuf.json_format import MessageToJson
from simple_di import Provide, inject
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from bentoml.exceptions import BentoMLException

from ..configuration import get_debug_mode
from ..configuration.containers import BentoMLContainer, BentoServerContainer
from ..server.instruments import InstrumentMiddleware
from ..types import HTTPRequest
from ..utils.open_api import get_open_api_spec_json
from .marshal.marshal import MARSHAL_REQUEST_HEADER, DataLoader

if TYPE_CHECKING:
    from bentoml._internal.service.inference_api import InferenceAPI
    from starlette.applications import Starlette

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
          url: '{url}',
          dom_id: '#swagger_ui_container'
      }})
  </script>
</body>
"""

SWAGGER_HTML = """\
<!DOCTYPE html>
<head>
  <link rel="stylesheet" type="text/css" href="static_content/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui-container"></div>
  <script src="static_content/swagger-ui-bundle.js"></script>
  <script>
      SwaggerUIBundle({{
          url: '{url}',
          dom_id: '#swagger-ui-container'
      }})
  </script>
</body>
"""


def _request_to_json(req):
    """
    Return request data for log prediction
    """
    if req.content_type == "application/json":
        return req.get_json()

    return {}


def log_exception(exc_info):
    """
    Logs an exception.  This is called by :meth:`handle_exception`
    if debugging is disabled and right before the handler is called.
    The default implementation logs the exception as error on the
    :attr:`logger`.
    """
    logger.error(
        "Exception on %s [%s]", request.path, request.method, exc_info=exc_info
    )


from bentoml._internal.server.base_app import BaseApp


class ServiceApp(BaseApp):
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
        enable_swagger: bool = Provide[
            BentoMLContainer.config.bento_server.swagger.enabled
        ],
        enable_metrics: bool = Provide[
            BentoMLContainer.config.bento_server.metrics.enabled
        ],
        tracer=Provide[BentoMLContainer.tracer],
    ):
        from starlette.applications import Starlette

        self.bento_service = bento_service

        self.app_name = bento_service.name
        self.enable_metrics = enable_metrics
        self.tracer = tracer
        self.enable_swagger = enable_swagger

        self.app = Starlette()
        for middleware in (InstrumentMiddleware,):  # TODO(jiang)
            self.app.add_middleware(middleware)

        for middleware, options in bento_service._middlewares:
            self.app.add_middleware(middleware, **options)

        for app, path in bento_service._mount_apps:
            self.app.mount(app=app, path=path)

    def setup(self) -> None:
        pass

    def index_view_func(self, request):
        """
        The index route for BentoML API server
        """
        from starlette.responses import HTMLResponse

        return HTMLResponse(DEFAULT_INDEX_HTML)

    def default_index_view_func(self):
        """
        The default index view for BentoML API server. This includes the readme
        generated from docstring and swagger UI
        """
        if not self.enable_swagger:
            return Response(
                content="Swagger is disabled", status_code=404, media_type="text/html"
            )
        return Response(
            content=DEFAULT_INDEX_HTML.format(
                url="/docs.json", readme=self.bento_service.__doc__
            ),
            status_code=200,
            media_type="text/html",
        )

    def swagger_ui_func(self):
        """
        The swagger UI route for BentoML API server
        """
        if not self.enable_swagger:
            return Response(
                content="Swagger is disabled", status_code=404, media_type="text/html"
            )
        return Response(
            content=SWAGGER_HTML.format(url="/docs.json"),
            status_code=200,
            media_type="text/html",
        )

    @staticmethod
    def docs_view_func(bento_service) -> "Response":
        docs = get_open_api_spec_json(bento_service)
        return JSONResponse(docs)

    @staticmethod
    def metadata_json_func(bento_service) -> "Response":
        bento_service_metadata = bento_service.get_bento_service_metadata_pb()
        return JSONResponse(MessageToJson(bento_service_metadata))

    def setup_routes(self):
        """
        Setup routes for bento model server, including:

        /               Index Page
        /docs           Swagger UI
        /healthz        Health check ping
        /feedback       Submitting feedback
        /metrics        Prometheus metrics endpoint
        /metadata       BentoService Artifact Metadata

        And user defined InferenceAPI list into flask routes, e.g.:
        /classify
        /predict
        """
        super().setup_routes()

        from starlette.staticfiles import StaticFiles

        self.app.add_route(path="/", name="home", route=self.index_view_func)
        self.app.add_route(path="/docs", name="swagger", route=self.swagger_ui_func)
        self.app.mount(
            path="/static_content",
            name="static_content",
            app=StaticFiles(directory=self.swagger_path),
        )

        self.app.add_route(
            path="/docs.json",
            name="docs",
            route=partial(self.docs_view_func, self.bento_service),
        )
        self.app.add_route(
            path="/metadata",
            name="metadata",
            route=partial(self.metadata_json_func, self.bento_service),
        )

        if self.enable_metrics:
            self.app.add_route(
                path="/metrics", name="metrics", route=self.metrics_view_func
            )

        self.setup_bento_service_api_routes()

    def setup_bento_service_api_routes(self):
        """
        Setup a route for each InferenceAPI object defined in bento_service
        """
        for _, api in self.bento_service._apis.items():
            route_function = self.get_api_route_function(api)
            self.app.add_route(
                path="/{}".format(api.route),
                name=api.name,
                route=route_function,
                methods=api.input.HTTP_METHODS,
            )

    def get_app(
        self,
        enable_access_control: bool = Provide[
            BentoMLContainer.config.bento_server.cors.enabled
        ],
        access_control_options: Dict = Provide[
            BentoServerContainer.access_control_options
        ],
    ) -> "ASGIApp":
        if enable_access_control:
            assert (
                access_control_options.get("access_control_allow_origin") is not None
            ), "To enable cors, access_control_allow_origin must be set"

            from starlette.middleware.cors import CORSMiddleware

            self.app.add_middleware(CORSMiddleware, **access_control_options)

        return self.app

    def get_api_route_function(
        self, api: "InferenceAPI"
    ) -> t.Callable[["Request"], t.Coroutine[t.Any, t.Any, "Response"]]:
        """
        Create api function for flask route, it wraps around user defined API
        callback and adapter class, and adds request logging and instrument metrics
        """

        async def api_func(
            request: "Request",
        ) -> "Response":
            # handle_request may raise 4xx or 5xx exception.
            try:
                input_data = api.input.from_http_request(request)
                output = api.func(input_data)
                response = await api.output.to_http_response(output)
            except BentoMLException as e:
                log_exception(sys.exc_info())

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
                log_exception(sys.exc_info())

                response = JSONResponse(
                    "An error has occurred in BentoML user code when handling this "
                    "request, find the error details in server logs",
                    status_code=500,
                )

            return response

        return api_func

        """
        def api_func_with_tracing():
            with self.tracer.span(
                service_name=f"BentoService.{self.bento_service.name}",
                span_name=f"InferenceAPI {api.name} HTTP route",
                request_headers=request.headers,
            ):
                return api_func()

        return api_func_with_tracing
        """

    @inject
    def metrics_view_func(self, client=Provide[BentoServerContainer.metrics_client]):
        return Response(
            client.generate_latest(),
            media_type=client.CONTENT_TYPE_LATEST,
        )
