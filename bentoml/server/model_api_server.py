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
"""
This module should generate a BentoModelApiServer class.
BentoModelApiServer takes in a list of models and serve their prediction from rest endpoints.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid
from time import time
from flask import Flask, url_for, jsonify, Response, request
from prometheus_client import generate_latest, Summary
from bentoml.server.prediction_logger import initialize_prediction_logger, log_prediction

CONTENT_TYPE_LATEST = str('text/plain; version=0.0.4; charset=utf-8')


def has_no_empty_params(rule):
    """
    return boolean if the rules have empty params or not
    """
    defaults = rule.defaults if rule.defaults is not None else ()
    arguments = rule.arguments if rule.arguments is not None else ()
    return len(defaults) >= len(arguments)


def create_api_function_wrapper(logger, model_name, model_version, api):
    """
    Create api function for flask route
    """
    summary_name = model_name + '_' + model_version + '_' + api.name
    request_metric_time = Summary(summary_name, summary_name + ' request latency')

    def wrapper():
        with request_metric_time.time():
            request_time = time()
            request_id = str(uuid.uuid4())
            response = api.handler.handle_request(request, api.func)
            response.headers['request_id'] = request_id
            if response.status_code == 200:
                metadata = {
                    'model_name': model_name,
                    'model_version': model_version,
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


class BentoModelApiServer():
    """
    Bento Model API Server defines how to start a REST API server with Bento Model.
    """

    def __init__(self, name, model_service, port=8000):
        self.port = port
        self.model_service = model_service

        self.app = Flask(name)
        self.prediction_logger = initialize_prediction_logger()

        self.setup_routes()

    def index_route(self):
        """
        The index route for bento model server, it display all avaliable routes
        """
        # TODO: Generate a html page for user and swagger definitions
        links = []
        for rule in self.app.url_map.iter_rules():
            if "GET" in rule.methods and has_no_empty_params(rule):
                url = url_for(rule.endpoint, **(rule.defaults or {}))
                links.append(url)
        response = jsonify(links=links)
        return response

    def healthz_route(self):
        """
        Health check for bento model server.
        Make sure it works with Kubernetes liveness probe
        """
        return Response(response='\n', status=200, mimetype='application/json')

    def setup_metrics_route(self):
        """
        Setup prometheus metrics routes.
        """

        def metrics_func():
            return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

        self.app.add_url_rule('/metrics', 'metrics', metrics_func)

    def feedback_route(self):
        """
        User send feedback along with the request Id. It will be stored and
        ready for further process.
        """

    def setup_api_func_route(self, model_name, model_version, route_name, api):
        """
        Setup a route for the api function from model service
        """
        route_function = create_api_function_wrapper(self.prediction_logger, model_name,
                                                     model_version, api)
        self.app.add_url_rule(rule='/' + route_name.replace('.', '/'), endpoint=route_name,
                              view_func=route_function, methods=['POST', 'GET'])

    def setup_endpoints(self):
        """
        Setup user defined endpoints into flask routes
        When there is only one api object on the model, we will create a /api_name route.
        If user defined more than 1 apis on the model, we will create routes in the format of
        model_name/model_version/api_name
        """
        model_service = self.model_service
        apis = self.model_service.get_service_apis()
        if len(apis) == 1:
            self.setup_api_func_route(model_service.name, model_service.version, apis[0].name,
                                      apis[0])
        else:
            for api in apis:
                route_name = model_service.name + '.' + model_service.version + '.' + api.name
                self.setup_api_func_route(model_service.name, model_service.version, route_name,
                                          api)

    def setup_routes(self):
        """
        Setup routes for bento model server.
        /, /healthz, /feedback, /metrics, /predict
        """

        self.app.add_url_rule('/', 'index', self.index_route)
        self.app.add_url_rule('/healthz', 'healthz', self.healthz_route)
        self.app.add_url_rule(rule='/feedback', endpoint='feedback', view_func=self.feedback_route,
                              methods=['POST', 'GET'])
        self.setup_metrics_route()
        self.setup_endpoints()

    def start(self, port=None):
        """
        Start an REST server at the specific port on the instance or parameter.
        """
        port = port if port is not None else self.port

        if self.model_service.loaded is True:
            self.app.run(port=port)
        else:
            raise Exception('model service is not loaded')
