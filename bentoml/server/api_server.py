# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import uuid
from functools import partial

from flask import (
    Flask,
    Response,
    jsonify,
    make_response,
    request,
    send_from_directory,
)
from werkzeug.exceptions import BadRequest, NotFound

from bentoml import BentoService, config
from bentoml.configuration import get_debug_mode
from bentoml.exceptions import BentoMLException
from bentoml.marshal.utils import DataLoader
from bentoml.server import trace
from bentoml.server.instruments import InstrumentMiddleware
from bentoml.server.open_api import get_open_api_spec_json
from bentoml.service import InferenceAPI

CONTENT_TYPE_LATEST = str("text/plain; version=0.0.4; charset=utf-8")

feedback_logger = logging.getLogger("bentoml.feedback")
logger = logging.getLogger(__name__)

INDEX_HTML = '''\
<!DOCTYPE html>
<head><link rel="stylesheet" type="text/css"
            href="/swagger_static/swagger-ui.css"></head>
<body>
<div id="swagger-ui-container"></div>
<script src="/swagger_static/swagger-ui-bundle.js"></script>
<script>
    SwaggerUIBundle({{
        url: '{url}',
        dom_id: '#swagger-ui-container'
    }})
</script>
</body>
'''


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


class BentoAPIServer:
    """
    BentoAPIServer creates a REST API server based on APIs defined with a BentoService
    via BentoService#get_service_apis call. Each InferenceAPI will become one
    endpoint exposed on the REST server, and the RequestHandler defined on each
    InferenceAPI object will be used to handle Request object before feeding the
    request data into a Service API function
    """

    DEFAULT_PORT = config("apiserver").getint("default_port")
    _MARSHAL_FLAG = config("marshal_server").get("marshal_request_header_flag")

    def __init__(self, bento_service: BentoService, port=DEFAULT_PORT, app_name=None):
        app_name = bento_service.name if app_name is None else app_name

        self.port = port
        self.bento_service = bento_service
        self.app = Flask(app_name, static_folder=None)
        self.static_path = self.bento_service.get_web_static_content_path()

        self.swagger_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'swagger_static'
        )

        for middleware in (InstrumentMiddleware,):
            self.app.wsgi_app = middleware(self.app.wsgi_app, self.bento_service)

        self.setup_routes()

    def start(self):
        """
        Start an REST server at the specific port on the instance or parameter.
        """
        # Bentoml api service is not thread safe.
        # Flask dev server enabled threaded by default, disable it.
        self.app.run(
            port=self.port, threaded=False, debug=get_debug_mode(), use_reloader=False,
        )

    @staticmethod
    def static_serve(static_path, file_path):
        """
        The static files route for BentoML API server
        """
        try:
            return send_from_directory(static_path, file_path)
        except NotFound:
            return send_from_directory(
                os.path.join(static_path, file_path), "index.html"
            )

    @staticmethod
    def index_view_func(static_path):
        """
        The index route for BentoML API server
        """
        return send_from_directory(static_path, 'index.html')

    @staticmethod
    def swagger_ui_func():
        """
        The swagger UI route for BentoML API server
        """
        return Response(
            response=INDEX_HTML.format(url='/docs.json'),
            status=200,
            mimetype="text/html",
        )

    @staticmethod
    def swagger_static(static_path, filename):
        """
        The swagger static files route for BentoML API server
        """
        return send_from_directory(static_path, filename)

    @staticmethod
    def docs_view_func(bento_service):
        docs = get_open_api_spec_json(bento_service)
        return jsonify(docs)

    @staticmethod
    def healthz_view_func():
        """
        Health check for BentoML API server.
        Make sure it works with Kubernetes liveness probe
        """
        return Response(response="\n", status=200, mimetype="text/plain")

    def metrics_view_func(self):
        # noinspection PyProtectedMember
        from prometheus_client import generate_latest

        return generate_latest()

    @staticmethod
    def feedback_view_func(bento_service):
        """
        User send feedback along with the request_id. It will be stored is feedback logs
        ready for further process.
        """
        data = request.get_json()

        if not data:
            raise BadRequest("Failed parsing feedback JSON data")

        if "request_id" not in data:
            raise BadRequest("Missing 'request_id' in feedback JSON data")

        data["service_name"] = bento_service.name
        data["service_version"] = bento_service.version
        feedback_logger.info(data)
        return "success"

    def setup_routes(self):
        """
        Setup routes for bento model server, including:

        /               Index Page
        /docs           Swagger UI
        /healthz        Health check ping
        /feedback       Submitting feedback
        /metrics        Prometheus metrics endpoint

        And user defined InferenceAPI list into flask routes, e.g.:
        /classify
        /predict
        """
        if self.static_path:
            # serve static files for any given path
            # this will also serve index.html from directory /any_path/
            # for path as /any_path/
            self.app.add_url_rule(
                "/<path:file_path>",
                "static_proxy",
                partial(self.static_serve, self.static_path),
            )
            # serve index.html from the directory /any_path
            # for path as /any_path/index
            self.app.add_url_rule(
                "/<path:file_path>/index",
                "static_proxy2",
                partial(self.static_serve, self.static_path),
            )
            # serve index.html from root directory for path as /
            self.app.add_url_rule(
                "/", "index", partial(self.index_view_func, self.static_path)
            )
        else:
            self.app.add_url_rule("/", "index", self.swagger_ui_func)

        self.app.add_url_rule("/docs", "swagger", self.swagger_ui_func)
        self.app.add_url_rule(
            "/swagger_static/<path:filename>",
            "swagger_static",
            partial(self.swagger_static, self.swagger_path),
        )
        self.app.add_url_rule(
            "/docs.json", "docs", partial(self.docs_view_func, self.bento_service)
        )
        self.app.add_url_rule("/healthz", "healthz", self.healthz_view_func)

        if config("apiserver").getboolean("enable_metrics"):
            self.app.add_url_rule("/metrics", "metrics", self.metrics_view_func)

        if config("apiserver").getboolean("enable_feedback"):
            self.app.add_url_rule(
                "/feedback",
                "feedback",
                partial(self.feedback_view_func, self.bento_service),
                methods=["POST"],
            )

        self.setup_bento_service_api_routes()

    def setup_bento_service_api_routes(self):
        """
        Setup a route for each InferenceAPI object defined in bento_service
        """
        for api in self.bento_service.inference_apis:
            route_function = self.bento_service_api_func_wrapper(api)
            self.app.add_url_rule(
                rule="/{}".format(api.name),
                endpoint=api.name,
                view_func=route_function,
                methods=api.input_adapter.HTTP_METHODS,
            )

    def bento_service_api_func_wrapper(self, api: InferenceAPI):
        """
        Create api function for flask route, it wraps around user defined API
        callback and adapter class, and adds request logging and instrument metrics
        """
        request_id = str(uuid.uuid4())

        def api_func():
            # handle_request may raise 4xx or 5xx exception.
            try:
                if request.headers.get(self._MARSHAL_FLAG):
                    reqs = DataLoader.split_requests(request.get_data())
                    responses = api.handle_batch_request(reqs)
                    response_body = DataLoader.merge_responses(responses)
                    response = make_response(response_body)
                else:
                    response = api.handle_request(request)
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
                    response = make_response('', e.status_code)
            except Exception:  # pylint: disable=broad-except
                # For all unexpected error, return 500 by default. For example,
                # if users' model raises an error of division by zero.
                log_exception(sys.exc_info())

                response = make_response(
                    'An error has occurred in BentoML user code when handling this '
                    'request, find the error details in server logs',
                    500,
                )

            response.headers["request_id"] = request_id

            return response

        def api_func_with_tracing():
            with trace(
                request.headers, service_name=self.__class__.__name__,
            ):
                return api_func()

        return api_func_with_tracing
