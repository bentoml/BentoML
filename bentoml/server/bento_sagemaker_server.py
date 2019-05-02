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

from flask import Flask, Response, request


def bento_sagemaker_api_wrapper(api):
    def wrapper():
        response = api.handle_request(request)
        return response
    return wrapper


def setup_bento_service_api_route(app, api):
    route_function = bento_sagemaker_api_wrapper(api)
    app.add_url_rule(rule='/invocations/{}'.format(api.name), endpoint=api.name,
                     view_func=route_function, methods=['POST'])


def ping_view_func():
    return Response(response='\n', status=200, mimetype='application/json')


def setup_routes(app, bento_service):
    """
    Setup routes required for AWS sagemaker
    /ping
    /invocation
    """
    app.add_url_rule('/ping', 'ping', ping_view_func)
    for api in bento_service.get_service_apis():
        setup_bento_service_api_route(app, api)


class BentoSagemakerServer():
    """
    BentoSagemakerServer create an AWS sagemaker compatibility reset server.
    """

    _DEFAULT_PORT = 8080

    def __init__(self, bento_service, app_name=None):
        app_name = bento_service.name if app_name is None else app_name

        self.bento_service = bento_service
        self.app = Flask(app_name)
        setup_routes(self.app, self.bento_service)

    def start(self):
        self.app.run(port=BentoSagemakerServer._DEFAULT_PORT)
