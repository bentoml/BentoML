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

from flask import Flask, Response, request
from bentoml.utils.usage_stats import track_server


def setup_bento_service_api_route(app, api):
    def view_function():
        return api.handle_request(request)

    app.add_url_rule(
        rule="/invocations",
        endpoint=api.name,
        view_func=view_function,
        methods=["POST"],
    )


def ping_view_func():
    return Response(response="\n", status=200, mimetype="application/json")


def setup_routes(app, bento_service, api_name):
    """
    Setup routes required for AWS sagemaker
    /ping
    /invocations
    """
    app.add_url_rule("/ping", "ping", ping_view_func)
    api = bento_service.get_service_api(api_name)
    setup_bento_service_api_route(app, api)


class BentoSagemakerServer:
    """
    BentoSagemakerServer create an AWS Sagemaker compatibility rest API server.
    """

    _DEFAULT_PORT = 8080

    def __init__(self, bento_service, api_name, app_name=None):
        app_name = bento_service.name if app_name is None else app_name

        self.bento_service = bento_service
        self.app = Flask(app_name)
        setup_routes(self.app, self.bento_service, api_name)

    def start(self):
        track_server('sagemaker')
        self.app.run(port=BentoSagemakerServer._DEFAULT_PORT)
