import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Dict

from google.protobuf.json_format import MessageToJson
from simple_di import Provide, inject
from starlette.requests import Request
from starlette.responses import Response

from bentoml.exceptions import BentoMLException

from ..configuration import get_debug_mode
from ..configuration.containers import BentoMLContainer
from ..server.instruments import InstrumentMiddleware
from ..types import HTTPRequest
from ..utils.open_api import get_open_api_spec_json
from .marshal.marshal import MARSHAL_REQUEST_HEADER, DataLoader

if TYPE_CHECKING:
    from ..service import InferenceAPI

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


class ServiceApp:
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
        bundle_path: str = Provide[BentoMLContainer.bundle_path],
        app_name: str = None,
        runner_name: str = None,
        enable_swagger: bool = Provide[
            BentoMLContainer.config.bento_server.swagger.enabled
        ],
        enable_metrics: bool = Provide[
            BentoMLContainer.config.bento_server.metrics.enabled
        ],
        tracer=Provide[BentoMLContainer.tracer],
    ):
        from starlette.applications import Starlette

        from bentoml.saved_bundle.loader import load_from_dir

        assert bundle_path, repr(bundle_path)

        self.bento_service = load_from_dir(bundle_path)
        self.runner = self.bento_service.get_runner_by_name(runner_name)
        self.app_name = runner_name

        self.app = Starlette()
        self.static_path = self.bento_service.get_web_static_content_path()
        self.enable_metrics = enable_metrics
        self.tracer = tracer

        for middleware in (InstrumentMiddleware,):  # TODO(jiang)
            self.app.add_middleware(middleware)

        self.setup_routes()

    @inject
    def run(
        self,
        port: int = Provide[BentoMLContainer.forward_port],
        host: str = Provide[BentoMLContainer.forward_host],
    ):
        """
        Start an REST server at the specific port on the instance or parameter.
        """
        # Bentoml api service is not thread safe.
        # Flask dev server enabled threaded by default, disable it.
        logger.info("Starting BentoML API server in development mode..")
        self.app.run(
            host=host,
            port=port,
            threaded=False,
            debug=get_debug_mode(),
            use_reloader=False,
        )

    def index_view_func(self, request):
        """
        The index route for BentoML API server
        """
        # TODO(jiang): no jinja
        from starlette.templating import Jinja2Templates

        templates = Jinja2Templates(directory=self.static_path)
        return templates.TemplateResponse("index.html", {"request": request})

    def default_index_view_func(self):
        """
        The default index view for BentoML API server. This includes the readme
        generated from docstring and swagger UI
        """
        if not self.enable_swagger:
            return Response(
                response="Swagger is disabled", status=404, mimetype="text/html"
            )
        return Response(
            response=DEFAULT_INDEX_HTML.format(
                url="/docs.json", readme=self.bento_service.__doc__
            ),
            status=200,
            mimetype="text/html",
        )

    def swagger_ui_func(self):
        """
        The swagger UI route for BentoML API server
        """
        if not self.enable_swagger:
            return Response(
                response="Swagger is disabled", status=404, mimetype="text/html"
            )
        return Response(
            response=SWAGGER_HTML.format(url="/docs.json"),
            status=200,
            mimetype="text/html",
        )

    @staticmethod
    def docs_view_func(bento_service):
        docs = get_open_api_spec_json(bento_service)
        return jsonify(docs)

    @staticmethod
    def metadata_json_func(bento_service):
        bento_service_metadata = bento_service.get_bento_service_metadata_pb()
        return jsonify(MessageToJson(bento_service_metadata))

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
        self.app.add_route(
            path="/healthz", name="healthz", route=self.healthz_view_func
        )  # TODO: readyz/ livez

        if self.enable_metrics:
            self.app.add_route(
                path="/metrics", name="metrics", route=self.metrics_view_func
            )

        self.setup_bento_service_api_routes()

    def setup_bento_service_api_routes(self):
        """
        Setup a route for each InferenceAPI object defined in bento_service
        """
        for api in self.bento_service._apis:
            route_function = self.bento_service_api_func_wrapper(api)
            self.app.add_route(
                path="/{}".format(api.route),
                name=api.name,
                route=route_function,
                methods=api.input_adapter.HTTP_METHODS,
            )

    def get_app(
        self,
        enable_access_control: bool = Provide[
            BentoMLContainer.config.bento_server.cors.enabled
        ],
        access_control_options: Dict = Provide[BentoMLContainer.access_control_options],
    ):
        if enable_access_control:
            assert (
                access_control_options.get("access_control_allow_origin") is not None
            ), "To enable cors, access_control_allow_origin must be set"

            from starlette.middleware.cors import CORSMiddleware

            self.app.add_middleware(CORSMiddleware, **access_control_options)

        return self.app

    def bento_service_api_func_wrapper(self, api: "InferenceAPI"):
        """
        Create api function for flask route, it wraps around user defined API
        callback and adapter class, and adds request logging and instrument metrics
        """

        def api_func():
            # handle_request may raise 4xx or 5xx exception.
            try:
                if request.headers.get(MARSHAL_REQUEST_HEADER):
                    reqs = DataLoader.split_requests(request.get_data())
                    responses = api.handle_batch_request(reqs)
                    response_body = DataLoader.merge_responses(responses)
                    response = make_response(response_body)
                else:
                    req = Request.from_flask_request(request)
                    resp = api.handle_request(req)
                    response = resp.to_flask_response()
            except BentoMLException as e:
                log_exception(sys.exc_info())

                if 400 <= e.status_code < 500 and e.status_code not in (401, 403):
                    response = make_response(
                        jsonify(
                            message="BentoService error handling API request: %s"
                            % str(e)
                        ),
                        e.status_code,
                    )
                else:
                    response = make_response("", e.status_code)
            except Exception:  # pylint: disable=broad-except
                # For all unexpected error, return 500 by default. For example,
                # if users' model raises an error of division by zero.
                log_exception(sys.exc_info())

                response = make_response(
                    "An error has occurred in BentoML user code when handling this "
                    "request, find the error details in server logs",
                    500,
                )

            return response

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
    def metrics_view_func(self, client=Provide[BentoMLContainer.metrics_client]):
        return Response(
            client.generate_latest(),
            media_type=client.CONTENT_TYPE_LATEST,
        )
