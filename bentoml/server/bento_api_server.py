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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import uuid
import json
import time
import logging
from functools import partial
from collections import OrderedDict

from flask import Flask, jsonify, Response, request, make_response
from werkzeug.utils import secure_filename

from bentoml import config
from bentoml.utils.usage_stats import track_server
from bentoml.exceptions import BentoMLException
from .middlewares import InstrumentMiddleware


CONTENT_TYPE_LATEST = str("text/plain; version=0.0.4; charset=utf-8")

prediction_logger = logging.getLogger("bentoml.prediction")
feedback_logger = logging.getLogger("bentoml.feedback")

logger = logging.getLogger(__name__)

INDEX_HTML = '''\
<!DOCTYPE html>
<head><link rel="stylesheet" type="text/css"
            href="/static/swagger-ui.css"></head>
<body>
<div id="swagger-ui-container"></div>
<script src="/static/swagger-ui-bundle.js"></script>
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


def get_docs(bento_service):
    """
    The docs for all endpoints in Open API format.
    """
    docs = OrderedDict(
        openapi="3.0.0",
        info=OrderedDict(
            version=bento_service.version,
            title=bento_service.name,
            description="To get a client SDK, copy all content from <a "
            "href=\"/docs.json\">docs</a> and paste into "
            "<a href=\"https://editor.swagger.io\">editor.swagger.io</a> then click "
            "the tab <strong>Generate Client</strong> and choose the language.",
        ),
        tags=[{"name": "infra"}, {"name": "app"}],
    )

    paths = OrderedDict()
    default_response = {"200": {"description": "success"}}

    paths["/healthz"] = OrderedDict(
        get=OrderedDict(
            tags=["infra"],
            description="Health check endpoint. Expecting an empty response with status"
            " code 200 when the service is in health state",
            responses=default_response,
        )
    )
    if config("apiserver").getboolean("enable_metrics"):
        paths["/metrics"] = OrderedDict(
            get=OrderedDict(
                tags=["infra"],
                description="Prometheus metrics endpoint",
                responses=default_response,
            )
        )
    if config("apiserver").getboolean("enable_feedback"):
        paths["/feedback"] = OrderedDict(
            get=OrderedDict(
                tags=["infra"],
                description="Predictions feedback endpoint. Expecting feedback request "
                "in JSON format and must contain a `request_id` field, which can be "
                "obtained from any BentoService API response header",
                responses=default_response,
            )
        )
        paths["/feedback"]["post"] = paths["/feedback"]["get"]

    for api in bento_service.get_service_apis():
        path = "/{}".format(api.name)
        paths[path] = OrderedDict(
            post=OrderedDict(
                tags=["app"],
                description=api.doc,
                requestBody=OrderedDict(required=True, content=api.request_schema),
                responses=default_response,
            )
        )

    docs["paths"] = paths
    return docs


class BentoAPIServer:
    """
    BentoAPIServer creates a REST API server based on APIs defined with a BentoService
    via BentoService#get_service_apis call. Each BentoServiceAPI will become one
    endpoint exposed on the REST server, and the RequestHandler defined on each
    BentoServiceAPI object will be used to handle Request object before feeding the
    request data into a Service API function
    """

    _DEFAULT_PORT = config("apiserver").getint("default_port")

    def __init__(self, bento_service, port=_DEFAULT_PORT, app_name=None):
        app_name = bento_service.name if app_name is None else app_name

        self.port = port
        self.bento_service = bento_service

        self.app = Flask(
            app_name,
            static_folder=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'static'
            ),
        )

        for middleware in (InstrumentMiddleware,):
            self.app.wsgi_app = middleware(self.app.wsgi_app, self.bento_service)

        self.setup_routes()

    def start(self):
        """
        Start an REST server at the specific port on the instance or parameter.
        """
        track_server('flask')

        self.app.run(port=self.port)

    @staticmethod
    def index_view_func():
        """
        The index route for BentoML API server
        """
        return Response(
            response=INDEX_HTML.format(url='/docs.json'),
            status=200,
            mimetype="text/html",
        )

    @staticmethod
    def docs_view_func(bento_service):
        docs = get_docs(bento_service)
        return jsonify(docs)

    @staticmethod
    def healthz_view_func():
        """
        Health check for BentoML API server.
        Make sure it works with Kubernetes liveness probe
        """
        return Response(response="\n", status=200, mimetype="application/json")

    def metrics_view_func(self):
        from prometheus_client import generate_latest

        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    @staticmethod
    def feedback_view_func(bento_service):
        """
        User send feedback along with the request Id. It will be stored and
        ready for further process.
        """
        if request.content_type != "application/json":
            return Response(
                response="Incorrect content format, require JSON", status=400
            )

        data = json.loads(request.data.decode("utf-8"))
        if "request_id" not in data.keys():
            return Response(response="Missing request id", status=400)

        if len(data.keys()) <= 1:
            return Response(response="Missing feedback data", status=400)

        data["service_name"] = bento_service.name
        data["service_version"] = bento_service.version

        feedback_logger.info(data)
        return Response(response="success", status=200)

    def setup_routes(self):
        """
        Setup routes for bento model server, including:

        /               Index Page
        /healthz        Health check ping
        /feedback       Submitting feedback
        /metrics        Prometheus metrics endpoint

        And user defined BentoServiceAPI list into flask routes, e.g.:
        /classify
        /predict
        """

        self.app.add_url_rule("/", "index", self.index_view_func)
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
                methods=["POST", "GET"],
            )

        self.setup_bento_service_api_routes()

    def setup_bento_service_api_routes(self):
        """
        Setup a route for each BentoServiceAPI object defined in bento_service
        """
        for api in self.bento_service.get_service_apis():
            route_function = self.bento_service_api_func_wrapper(api)
            self.app.add_url_rule(
                rule="/{}".format(api.name),
                endpoint=api.name,
                view_func=route_function,
                methods=api.handler.HTTP_METHODS,
            )

    @staticmethod
    def log_image(req, request_id):
        if not config('logging').getboolean('log_request_image_files'):
            return []

        img_prefix = 'image/'
        log_folder = config('logging').get('base_log_dir')

        all_paths = []

        if req.content_type and req.content_type.startswith(img_prefix):
            filename = '{timestamp}-{request_id}.{ext}'.format(
                timestamp=int(time.time()),
                request_id=request_id,
                ext=req.content_type[len(img_prefix) :],
            )
            path = os.path.join(log_folder, filename)
            all_paths.append(path)
            with open(path, 'wb') as f:
                f.write(req.get_data())

        for name in req.files:
            file = req.files[name]
            if file and file.filename:
                orig_filename = secure_filename(file.filename)
                filename = '{timestamp}-{request_id}-{orig_filename}'.format(
                    timestamp=int(time.time()),
                    request_id=request_id,
                    orig_filename=orig_filename,
                )
                path = os.path.join(log_folder, filename)
                all_paths.append(path)
                file.save(path)
                file.stream.seek(0)

        return all_paths

    def bento_service_api_func_wrapper(self, api):
        """
        Create api function for flask route, it wraps around user defined API
        callback and BentoHandler class, and adds request logging and instrument metrics
        """
        request_id = str(uuid.uuid4())
        service_name = self.bento_service.name
        service_version = self.bento_service.version

        def api_func_wrapper():
            # Log image files in request if there is any
            image_paths = self.log_image(request, request_id)

            # _request_to_json parses request as JSON; in case errors, it raises
            # a 400 exception. (consider 4xx before 5xx.)
            request_for_log = _request_to_json(request)

            # handle_request may raise 4xx or 5xx exception.
            try:
                response = api.handle_request(request)
            except BentoMLException as e:
                self.log_exception(sys.exc_info())

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
                self.log_exception(sys.exc_info())

                response = make_response(
                    'An error has occurred in BentoML user code when handling this '
                    'request, find the error details in server logs',
                    500,
                )

            request_log = {
                "request_id": request_id,
                "service_name": service_name,
                "service_version": service_version,
                "api": api.name,
                "request": request_for_log,
                "response_code": response.status_code,
            }

            if len(image_paths) > 0:
                request_log['image_paths'] = image_paths

            if 200 <= response.status_code < 300:
                request_log['response'] = response.response

            prediction_logger.info(request_log)

            response.headers["request_id"] = request_id

            return response

        return api_func_wrapper

    def log_exception(self, exc_info):
        """Logs an exception.  This is called by :meth:`handle_exception`
        if debugging is disabled and right before the handler is called.
        The default implementation logs the exception as error on the
        :attr:`logger`.
        """
        logger.error(
            "Exception on %s [%s]", request.path, request.method, exc_info=exc_info
        )
