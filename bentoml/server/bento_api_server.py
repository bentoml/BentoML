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
from flask import Flask, url_for, jsonify, Response, request
from prometheus_client import generate_latest, Summary
from bentoml.server.prediction_logger import initialize_prediction_logger, log_prediction
from bentoml.server.feedback_logger import initialize_feedback_logger, log_feedback

CONTENT_TYPE_LATEST = str('text/plain; version=0.0.4; charset=utf-8')


def has_empty_params(rule):
    """
    return True if the rule has empty params
    """
    defaults = rule.defaults if rule.defaults is not None else ()
    arguments = rule.arguments if rule.arguments is not None else ()
    return len(defaults) < len(arguments)


def bento_service_api_wrapper(api, service_name, service_version, logger):
    """
    Create api function for flask route
    """
    summary_name = str(service_name) + '_' + str(service_version) + '_' + str(api.name)
    request_metric_time = Summary(summary_name, summary_name + ' request latency')

    def wrapper():
        with request_metric_time.time():
            request_time = time()
            request_id = str(uuid.uuid4())
            response = api.handler.handle_request(request, api.func, api.options)
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


class BentoAPIServer():
    """
    BentoAPIServer creates a REST API server based on APIs defined with a BentoService
    via BentoService#get_service_apis call. Each BentoServiceAPI will become one
    endpoint exposed on the REST server, and the RequestHandler defined on each
    BentoServiceAPI object will be used to handle Request object before feeding the
    request data into a Service API function
    """

    def __init__(self, name, bento_service, port=8000):
        self.port = port
        self.bento_service = bento_service

        self.app = Flask(name)
        self.prediction_logger = initialize_prediction_logger()
        self.feedback_logger = initialize_feedback_logger()

        self._setup_routes()

    def _index_view_func(self):
        """
        The index route for bento model server, it display all avaliable routes
        """
        # TODO: Generate a html page for user and swagger definitions
        links = []
        for rule in self.app.url_map.iter_rules():
            if "GET" in rule.methods and not has_empty_params(rule):
                url = url_for(rule.endpoint, **(rule.defaults or {}))
                links.append(url)
        response = jsonify(links=links)
        return response

    def _healthz_view_func(self):
        """
        Health check for bento model server.
        Make sure it works with Kubernetes liveness probe
        """
        return Response(response='\n', status=200, mimetype='application/json')

    @staticmethod
    def _metrics_view_func():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    def _feedback_view_func(self):
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

        log_feedback(self.feedback_logger, data)
        return Response(response='success', status=200)

    def _setup_bento_service_api_route(self, bento_service, api):
        """
        Setup a route for one BentoServiceAPI object defined in bento_service
        """
        route_function = bento_service_api_wrapper(api, bento_service.name, bento_service.version,
                                                   self.prediction_logger)

        self.app.add_url_rule(rule='/{}'.format(api.name), endpoint=api.name,
                              view_func=route_function, methods=['POST', 'GET'])

    def _setup_routes(self):
        """
        Setup routes for bento model server.
        /, /healthz, /feedback, /metrics, /predict

        And user defined BentoServiceAPI list into flask routes
        """

        self.app.add_url_rule('/', 'index', self._index_view_func)
        self.app.add_url_rule('/healthz', 'healthz', self._healthz_view_func)
        self.app.add_url_rule('/feedback', 'feedback', self._feedback_view_func,
                              methods=['POST', 'GET'])
        self.app.add_url_rule('/metrics', 'metrics', BentoAPIServer._metrics_view_func)

        for api in self.bento_service.get_service_apis():
            self._setup_bento_service_api_route(self.bento_service, api)

    def start(self, port=None):
        """
        Start an REST server at the specific port on the instance or parameter.
        """
        port = port if port is not None else self.port
        self.app.run(port=port)
