# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid
import json
from time import time
from functools import partial

from flask import Flask, url_for, jsonify, Response, request
from prometheus_client import generate_latest, Summary

from bentoml.server.prediction_logger import get_prediction_logger, log_prediction
from bentoml.server.feedback_logger import get_feedback_logger, log_feedback

CONTENT_TYPE_LATEST = str('text/plain; version=0.0.4; charset=utf-8')

prediction_logger = get_prediction_logger()
feedback_logger = get_feedback_logger()


def has_empty_params(rule):
    """
    return True if the rule has empty params
    """
    defaults = rule.defaults if rule.defaults is not None else ()
    arguments = rule.arguments if rule.arguments is not None else ()
    return len(defaults) < len(arguments)


def index_view_func(app):
    """
    The index route for bento model server, it display all avaliable routes
    """
    # TODO: Generate a html page for user and swagger definitions
    links = []
    for rule in app.url_map.iter_rules():
        if "GET" in rule.methods and not has_empty_params(rule):
            url = url_for(rule.endpoint, **(rule.defaults or {}))
            links.append(url)

    return jsonify(links=links)


def healthz_view_func():
    """
    Health check for bento model server.
    Make sure it works with Kubernetes liveness probe
    """
    return Response(response='\n', status=200, mimetype='application/json')


def metrics_view_func():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


def feedback_view_func():
    """
    User send feedback along with the request Id. It will be stored and
    ready for further process.
    """
    if request.content_type != 'application/json':
        return Response(response='Incorrect content format, require JSON', status=400)

    data = json.loads(request.data.decode('utf-8'))
    if 'request_id' not in data.keys():
        return Response(response='Missing request id', status=400)

    if len(data.keys()) <= 1:
        return Response(response='Missing feedback data', status=400)

    log_feedback(feedback_logger, data)
    return Response(response='success', status=200)


def bento_service_api_wrapper(api, service_name, service_version, logger):
    """
    Create api function for flask route
    """
    summary_name = str(service_name) + '_' + str(api.name)
    request_metric_time = Summary(summary_name, summary_name + ' request latency')

    def wrapper():
        with request_metric_time.time():
            request_time = time()
            request_id = str(uuid.uuid4())
            response = api.handle_request(request)
            response.headers['request_id'] = request_id
            if response.status_code == 200:
                metadata = {
                    'service_name': service_name,
                    'service_version': service_version,
                    'api_name': api.name,
                    'request_id': request_id,
                    'asctime': request_time,
                }
                log_prediction(
                    logger,
                    metadata,
                    request,
                    response,
                )
            else:
                # TODO: log errors as well.
                pass

            return response

    return wrapper


def setup_bento_service_api_route(app, bento_service, api):
    """
    Setup a route for one BentoServiceAPI object defined in bento_service
    """
    route_function = bento_service_api_wrapper(api, bento_service.name, bento_service.version,
                                               prediction_logger)

    app.add_url_rule(rule='/{}'.format(api.name), endpoint=api.name, view_func=route_function,
                     methods=['POST', 'GET'])


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

    app.add_url_rule('/', 'index', partial(index_view_func, app))
    app.add_url_rule('/healthz', 'healthz', healthz_view_func)
    app.add_url_rule('/feedback', 'feedback', feedback_view_func, methods=['POST', 'GET'])
    app.add_url_rule('/metrics', 'metrics', metrics_view_func)

    for api in bento_service.get_service_apis():
        setup_bento_service_api_route(app, bento_service, api)


class BentoAPIServer():
    """
    BentoAPIServer creates a REST API server based on APIs defined with a BentoService
    via BentoService#get_service_apis call. Each BentoServiceAPI will become one
    endpoint exposed on the REST server, and the RequestHandler defined on each
    BentoServiceAPI object will be used to handle Request object before feeding the
    request data into a Service API function
    """

    _DEFAULT_PORT = 5000

    def __init__(self, bento_service, port=_DEFAULT_PORT, app_name=None):
        app_name = bento_service.name if app_name is None else app_name

        self.port = port
        self.bento_service = bento_service

        self.app = Flask(app_name)
        setup_routes(self.app, self.bento_service)

    def start(self):
        """
        Start an REST server at the specific port on the instance or parameter.
        """
        self.app.run(port=self.port)
