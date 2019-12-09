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
import uuid
import json
import time
import logging
from functools import partial
from collections import OrderedDict

from flask import Flask, jsonify, Response, request
from werkzeug.utils import secure_filename
from prometheus_client import generate_latest, Summary, Counter

from bentoml import config
from bentoml.utils.usage_stats import track_server


CONTENT_TYPE_LATEST = str("text/plain; version=0.0.4; charset=utf-8")

prediction_logger = logging.getLogger("bentoml.prediction")
feedback_logger = logging.getLogger("bentoml.feedback")

LOG = logging.getLogger(__name__)

INDEX_HTML = '''
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


def has_empty_params(rule):
    """
    return True if the rule has empty params
    """
    defaults = rule.defaults if rule.defaults is not None else ()
    arguments = rule.arguments if rule.arguments is not None else ()
    return len(defaults) < len(arguments)


def index_view_func():
    """
    The index route for bento model server
    """
    return Response(
        response=INDEX_HTML.format(url='/docs.json'), status=200, mimetype="text/html"
    )


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


def docs_view_func(bento_service):
    docs = get_docs(bento_service)
    return jsonify(docs)


def healthz_view_func():
    """
    Health check for bento model server.
    Make sure it works with Kubernetes liveness probe
    """
    return Response(response="\n", status=200, mimetype="application/json")


def metrics_view_func():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


def feedback_view_func(bento_service):
    """
    User send feedback along with the request Id. It will be stored and
    ready for further process.
    """
    if request.content_type != "application/json":
        return Response(response="Incorrect content format, require JSON", status=400)

    data = json.loads(request.data.decode("utf-8"))
    if "request_id" not in data.keys():
        return Response(response="Missing request id", status=400)

    if len(data.keys()) <= 1:
        return Response(response="Missing feedback data", status=400)

    data["service_name"] = bento_service.name
    data["service_version"] = bento_service.version

    feedback_logger.info(data)
    return Response(response="success", status=200)


def bento_service_api_wrapper(api, service_name, service_version):
    """
    Create api function for flask route
    """
    metric_name = '{}_{}'.format(service_name, api.name)
    namespace = config('instrument').get('default_namespace')

    request_duration = Summary(
        name=metric_name + '_request_duration_seconds',
        documentation=metric_name + " request duration in seconds",
        namespace=namespace,
    )
    request_counter = Counter(
        name=metric_name + "_counter",
        documentation='request count by response http status code',
        labelnames=['http_response_code'],
    )

    def log_image(req, request_id):
        img_prefix = 'image/'
        log_folder = config('logging').get('base_log_dir')

        all_paths = []

        if req.content_type.startswith(img_prefix):
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

    def wrapper():
        with request_duration.time():
            request_id = str(uuid.uuid4())
            # Assume there is not a strong use case for idempotency check here.
            # Will revise later if we find a case.

            image_paths = []
            if not config('logging').getboolean('disable_logging_image'):
                image_paths = log_image(request, request_id)

            response = api.handle_request(request)

            request_log = {
                "request_id": request_id,
                "service_name": service_name,
                "service_version": service_version,
                "api": api.name,
                "request": _request_to_json(request),
                "response_code": response.status_code,
            }

            if len(image_paths) > 0:
                request_log['image_paths'] = image_paths

            if 200 <= response.status_code < 300:
                request_log['response'] = response.response

            prediction_logger.info(request_log)

            response.headers["request_id"] = request_id

            # instrument request count by status_code
            request_counter.labels(response.status_code).inc()

            return response

        return response

    return wrapper


def setup_bento_service_api_route(app, bento_service, api):
    """
    Setup a route for one BentoServiceAPI object defined in bento_service
    """
    route_function = bento_service_api_wrapper(
        api, bento_service.name, bento_service.version
    )

    app.add_url_rule(
        rule="/{}".format(api.name),
        endpoint=api.name,
        view_func=route_function,
        methods=["POST", "GET"],
    )


def setup_routes(app, bento_service):
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

    app.add_url_rule("/", "index", index_view_func)
    app.add_url_rule("/docs.json", "docs", partial(docs_view_func, bento_service))
    app.add_url_rule("/healthz", "healthz", healthz_view_func)

    if config("apiserver").getboolean("enable_metrics"):
        app.add_url_rule("/metrics", "metrics", metrics_view_func)

    if config("apiserver").getboolean("enable_feedback"):
        app.add_url_rule(
            "/feedback",
            "feedback",
            partial(feedback_view_func, bento_service),
            methods=["POST", "GET"],
        )

    for api in bento_service.get_service_apis():
        setup_bento_service_api_route(app, bento_service, api)


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
        setup_routes(self.app, self.bento_service)

    def start(self):
        """
        Start an REST server at the specific port on the instance or parameter.
        """
        track_server('flask')

        self.app.run(port=self.port)
