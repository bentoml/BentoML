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
from functools import partial

from dependency_injector.wiring import Provide, inject
from flask import Flask, Response, jsonify, make_response, request, send_from_directory
from google.protobuf.json_format import MessageToJson
from werkzeug.exceptions import BadRequest, NotFound

from bentoml import BentoService
from bentoml.configuration import get_debug_mode
from bentoml.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException
from bentoml.marshal.utils import DataLoader
from bentoml.server.instruments import InstrumentMiddleware
from bentoml.server.open_api import get_open_api_spec_json
from bentoml.service import InferenceAPI
from bentoml.tracing import get_tracer

import requests
from datetime import datetime
import pytz
import numpy as np

CONTENT_TYPE_LATEST = str("text/plain; version=0.0.4; charset=utf-8")

feedback_logger = logging.getLogger("bentoml.feedback")
logger = logging.getLogger(__name__)


DEFAULT_INDEX_HTML = '''\
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
'''

SWAGGER_HTML = '''\
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

    @inject
    def __init__(
        self,
        bento_service: BentoService,
        app_name: str = None,
        enable_swagger: bool = Provide[
            BentoMLContainer.config.api_server.enable_swagger
        ],
        enable_metrics: bool = Provide[
            BentoMLContainer.config.api_server.enable_metrics
        ],
        enable_feedback: bool = Provide[
            BentoMLContainer.config.api_server.enable_feedback
        ],
        request_header_flag: str = Provide[
            BentoMLContainer.config.marshal_server.request_header_flag
        ],
    ):
        app_name = bento_service.name if app_name is None else app_name

        self.bento_service = bento_service
        self.app = Flask(app_name, static_folder=None)
        self.static_path = self.bento_service.get_web_static_content_path()
        self.enable_swagger = enable_swagger
        self.enable_metrics = enable_metrics
        self.enable_feedback = enable_feedback
        self.request_header_flag = request_header_flag

        self.swagger_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'static_content'
        )

        for middleware in (InstrumentMiddleware,):
            self.app.wsgi_app = middleware(self.app.wsgi_app, self.bento_service)

        self.setup_routes()

    def start(self, port: int, host: str = "127.0.0.1"):
        """
        Start an REST server at the specific port on the instance or parameter.
        """
        # Bentoml api service is not thread safe.
        # Flask dev server enabled threaded by default, disable it.
        self.app.run(
            host=host,
            port=port,
            threaded=False,
            debug=get_debug_mode(),
            use_reloader=False,
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
                url='docs.json', readme=self.bento_service.__doc__
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
            response=SWAGGER_HTML.format(url='docs.json'),
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

    @staticmethod
    def metadata_json_func(bento_service):
        bento_service_metadata = bento_service.get_bento_service_metadata_pb()
        return jsonify(MessageToJson(bento_service_metadata))

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
        /metadata       BentoService Artifact Metadata

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
            self.app.add_url_rule("/", "index", self.default_index_view_func)

        self.app.add_url_rule("/docs", "swagger", self.swagger_ui_func)
        self.app.add_url_rule(
            "/static_content/<path:filename>",
            "static_content",
            partial(self.swagger_static, self.swagger_path),
        )
        self.app.add_url_rule(
            "/docs.json", "docs", partial(self.docs_view_func, self.bento_service)
        )
        self.app.add_url_rule("/healthz", "healthz", self.healthz_view_func)
        self.app.add_url_rule(
            "/metadata",
            "metadata",
            partial(self.metadata_json_func, self.bento_service),
        )

        if self.enable_metrics:
            self.app.add_url_rule("/metrics", "metrics", self.metrics_view_func)

        if self.enable_feedback:
            self.app.add_url_rule(
                "/feedback",
                "feedback",
                partial(self.feedback_view_func, self.bento_service),
                methods=["POST"],
            )

        self.setup_bento_service_api_routes()

        self.app.add_url_rule(
            rule="/ngsi-ld/ml/processing",
            endpoint="processing",
            view_func=self.handle_ml_processing,
            methods=["POST"]
        )

        self.app.add_url_rule(
            rule="/ngsi-ld/ml/predict",
            endpoint="ml-predict",
            view_func=self.handle_ml_predict,
            methods=["POST"]
        )

    def handle_ml_processing(self):
        """
        Handle receipt of a notification from subscription to
        MLProcessing entities. It indicates a new entity is interested
        in using this MLModel.
        On receipt of this notification, the information on where to
        find the input data for prediction is retrieve, and a subscription
        is created to be notified when input data changes

        The notification received looks like:

        {
            'id': 'urn:ngsi-ld:Notification:fadc5090-2425-42f8-b318-1966fa0e0011',
            'type': 'Notification',
            'subscriptionId': 'urn:ngsi-ld:Subscription:MLModel:flow:predict:71dba318-2989-4c76-a22c-52a53f04759b',
            'notifiedAt': '2021-05-03T09:53:50.330686Z',
            'data': [
                {
                    'id': 'urn:ngsi-ld:MLProcessing:4bbb2b09-ad6c-4fb9-8f40-8d37e4cddd3a',
                    'type': 'MLProcessing',
                    'refSubscriptionQuery':
                        {
                            'type': 'Relationship',
                            'object': 'urn:ngsi-ld:MLProcessing:SubscriptionQuery:e7be459e-dcee-46ab-90da-fba3120db4ff'
                        },
                    '@context': [
                        'https://raw.githubusercontent.com/easy-global-market/ngsild-api-data-models/master/mlaas/jsonld-contexts/mlaas-compound.jsonl'
                    ]
                }
            ]
        }

        We need to:
        * GET the SubscriptionQuery entity referenced by 'refSubscriptionQuery',
        * extract from the SubscriptionQuery entity, where to get the input data,
        * Finally create a subscription to this data. 
        """
        # Hard coded notification for now ...
        # Some generic configuration
        access_token = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJIS0xvcnlIcDN6VHlOY0EtNWYwQ19iVC1hbm5ldDFLSUhFT0U1VnN1NjRrIn0.eyJleHAiOjE2MjA2MzE2MDIsImlhdCI6MTYxOTc2NzYwMiwianRpIjoiYThhODhhOGMtM2Q5ZS00ZjEyLWFmNzQtMzNiNzM3ZmVlODAxIiwiaXNzIjoiaHR0cHM6Ly9zc28uZWdsb2JhbG1hcmsuY29tL2F1dGgvcmVhbG1zL3N0ZWxsaW8tZGV2IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6ImNhNDQxNzE0LTk3YjYtNGJiOS04Y2E3LTBiMjM5NDA0NDNmYSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImRpZGllci1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiNGQxNjgzNGEtNDU5My00YTc0LThjZDEtODYzZTU4OGFmODcyIiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJzdGVsbGlvLWNyZWF0b3IiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiZGlkaWVyLWNsaWVudCI6eyJyb2xlcyI6WyJ1bWFfcHJvdGVjdGlvbiJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGVtYWlsIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJjbGllbnRJZCI6ImRpZGllci1jbGllbnQiLCJjbGllbnRIb3N0IjoiOTIuMTg0LjEwMi4yNDkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJzZXJ2aWNlLWFjY291bnQtZGlkaWVyLWNsaWVudCIsImNsaWVudEFkZHJlc3MiOiI5Mi4xODQuMTAyLjI0OSJ9.nWv9vQn0Gt6MnJYsdReSOpQUOweLIKcCZknuauQuWKI3UCqEpLcTLB6M175rUGAauiPe8IIdCnEENEnFOv_kri9tYa5RujKlgLxxoSKqjjV_G-E64VZKlKTi0iPLVm4thRsHRXl_DH-vaoQ8u53MiaHSIqY4IdXcvpNgLTxhNjPwlx887I8MZFDbn63a15NWM3QPbhOBkYnOdnCHif-CVRGu2e0aJ4Dk4g36e_au7DtWCXl7Kia0vR5_14wLUoWftyyjeKK7uelcLJs3eSNWXcjFnYgc9aYSD2I9rVahYZQmm8Niaqtp1DFEXCUrGFiZ9lNExOIItTdY8FJRsKR9zA'
        headers = {
            'Authorization': 'Bearer ' + access_token,
            'Content-Type': 'application/ld+json'
        }
        URL_ENTITIES = 'https://stellio-dev.eglobalmark.com/ngsi-ld/v1/entities/'
        URL_SUBSCRIPTION = 'https://stellio-dev.eglobalmark.com/ngsi-ld/v1/subscriptions/'
        SUBSCRIPTION_INPUT_DATA = 'urn:ngsi-ld:Subscription:input:data:2c30fa86-a25c-4191-8311-8954294e92b3'
        ENDPOINT_INPUT_DATA_CHANGED = 'https://0ba2eb3a-2ff5-4a72-9a6f-f430f9f41ad3.mock.pstmn.io/mlprocessing'
        AT_CONTEXT = [
            'https://raw.githubusercontent.com/easy-global-market/ngsild-api-data-models/master/mlaas/jsonld-contexts/mlaas-precipitation-contexts.jsonld'
        ]

        mlprocessing_notification = {
            'id': 'urn:ngsi-ld:Notification:fadc5090-2425-42f8-b318-1966fa0e0011',
            'type': 'Notification',
            'subscriptionId': 'urn:ngsi-ld:Subscription:MLModel:flow:predict:71dba318-2989-4c76-a22c-52a53f04759b',
            'notifiedAt': '2021-05-03T09:53:50.330686Z',
            'data': [
                {
                    'id': 'urn:ngsi-ld:MLProcessing:4bbb2b09-ad6c-4fb9-8f40-8d37e4cddd3a',
                    'type': 'MLProcessing',
                    'refSubscriptionQuery':
                        {
                            'type': 'Relationship',
                            'object': 'urn:ngsi-ld:MLProcessing:SubscriptionQuery:e7be459e-dcee-46ab-90da-fba3120db4ff'
                        },
                    '@context': [
                        'https://raw.githubusercontent.com/easy-global-market/ngsild-api-data-models/master/mlaas/jsonld-contexts/mlaas-compound.jsonl'
                    ]
                }
            ]
        }

        # Getting the SubscriptionQuery entity
        refSubscriptionQuery = mlprocessing_notification['data'][0]['refSubscriptionQuery']['object']
        r = requests.get(URL_ENTITIES+refSubscriptionQuery, headers=headers)
        print(f'requests status_code GET subscriptionQuery: {r.status_code}')
        print('r.json()')
        print(r.json())
        ENTITY_INPUT_DATA = r.json()['entityID']['value']

        # We don't really use the content of the SubscriptionQuery entity for now ...
        # Instead we build our own 'hard coded' Subscription to input data
        json_ = {
            '@context': AT_CONTEXT,
            'id':SUBSCRIPTION_INPUT_DATA,
            'type':'Subscription',
            'entities': [
                {
                    'id': ENTITY_INPUT_DATA,
                    'type': 'River'
                }
            ],
            'watchedAttributes': ['precipitation'],
            'notification': {
                'endpoint': {
                    'uri': ENDPOINT_INPUT_DATA_CHANGED,
                    'accept': 'application/json'
                },
                'attributes': ['precipitation']
            }
        }

        print('Creating subscription to precipitation')
        # Creating the subscription to precipitation
        r = requests.post(URL_SUBSCRIPTION, json=json_, headers=headers)
        print(f'requests status_code POST Subscription Precipitation: {r.status_code}')

        # Finally, respond to the initial received request (notification)
        # with empty 200        
        response = make_response(
            '',
            200,
        )
        return response

    def handle_ml_predict(self):
        """
        Handle the request for a prediction. The request is actually a NGSI-LD
        notification of the change of a particular property of an NGSI-LD
        Entity.

        The notification received looks like:

        {
            "id": "urn:ngsi-ld:Notification:cc231a15-d220-403c-bfc6-ad60bc49466f",
            "type": "Notification",
            "subscriptionId": "urn:ngsi-ld:Subscription:input:data:2c30fa86-a25c-4191-8311-8954294e92b3",
            "notifiedAt": "2021-05-04T06:45:32.83178Z",
            "data": [
                {
                "id": "urn:ngsi-ld:River:014f5730-72ab-4554-a106-afbe5d4d9d26",
                "type": "River",
                "precipitation": {
                    "type": "Property",
                    "createdAt": "2021-05-04T06:45:32.674520Z",
                    "value": 2.2,
                    "observedAt": "2021-05-04T06:35:22.000Z",
                    "unitCode": "MMT"
                },
                "@context": [
                    "https://raw.githubusercontent.com/easy-global-market/ngsild-api-data-models/master/mlaas/jsonld-contexts/mlaas-precipitation-compound.jsonld"
                ]
                }
            ]
        }

        We need to:
        * Extract the input_data from the NGSI-LD Notification,
        * Reshape the data (2 dims array)
        * Predict using the deployed BentoML service at /predict
        * Create a NGSI-LD request to update the appropriate Entity/Property
        * Update the Entity/Property via PATCH
        """
        # Some generic configuration
        access_token = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJIS0xvcnlIcDN6VHlOY0EtNWYwQ19iVC1hbm5ldDFLSUhFT0U1VnN1NjRrIn0.eyJleHAiOjE2MjA2MzE2MDIsImlhdCI6MTYxOTc2NzYwMiwianRpIjoiYThhODhhOGMtM2Q5ZS00ZjEyLWFmNzQtMzNiNzM3ZmVlODAxIiwiaXNzIjoiaHR0cHM6Ly9zc28uZWdsb2JhbG1hcmsuY29tL2F1dGgvcmVhbG1zL3N0ZWxsaW8tZGV2IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6ImNhNDQxNzE0LTk3YjYtNGJiOS04Y2E3LTBiMjM5NDA0NDNmYSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImRpZGllci1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiNGQxNjgzNGEtNDU5My00YTc0LThjZDEtODYzZTU4OGFmODcyIiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJzdGVsbGlvLWNyZWF0b3IiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiZGlkaWVyLWNsaWVudCI6eyJyb2xlcyI6WyJ1bWFfcHJvdGVjdGlvbiJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGVtYWlsIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJjbGllbnRJZCI6ImRpZGllci1jbGllbnQiLCJjbGllbnRIb3N0IjoiOTIuMTg0LjEwMi4yNDkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJzZXJ2aWNlLWFjY291bnQtZGlkaWVyLWNsaWVudCIsImNsaWVudEFkZHJlc3MiOiI5Mi4xODQuMTAyLjI0OSJ9.nWv9vQn0Gt6MnJYsdReSOpQUOweLIKcCZknuauQuWKI3UCqEpLcTLB6M175rUGAauiPe8IIdCnEENEnFOv_kri9tYa5RujKlgLxxoSKqjjV_G-E64VZKlKTi0iPLVm4thRsHRXl_DH-vaoQ8u53MiaHSIqY4IdXcvpNgLTxhNjPwlx887I8MZFDbn63a15NWM3QPbhOBkYnOdnCHif-CVRGu2e0aJ4Dk4g36e_au7DtWCXl7Kia0vR5_14wLUoWftyyjeKK7uelcLJs3eSNWXcjFnYgc9aYSD2I9rVahYZQmm8Niaqtp1DFEXCUrGFiZ9lNExOIItTdY8FJRsKR9zA'
        headers = {
            'Authorization': 'Bearer ' + access_token,
            'Content-Type': 'application/ld+json'
        }
        RIVER_SIAGNE_UUID = 'urn:ngsi-ld:River:014f5730-72ab-4554-a106-afbe5d4d9d26'
        MLMODEL_UUID = 'urn:ngsi-ld:MLModel:flow:predict'
        URL_ENTITIES = 'https://stellio-dev.eglobalmark.com/ngsi-ld/v1/entities/'
        AT_CONTEXT = [
            'https://raw.githubusercontent.com/easy-global-market/ngsild-api-data-models/master/mlaas/jsonld-contexts/mlaas-precipitation-contexts.jsonld'
        ]

        # Extract input data from NGSLI-LD notification
        input_data_notification = {
            "id": "urn:ngsi-ld:Notification:cc231a15-d220-403c-bfc6-ad60bc49466f",
            "type": "Notification",
            "subscriptionId": "urn:ngsi-ld:Subscription:input:data:2c30fa86-a25c-4191-8311-8954294e92b3",
            "notifiedAt": "2021-05-04T06:45:32.83178Z",
            "data": [
                {
                    "id": "urn:ngsi-ld:River:014f5730-72ab-4554-a106-afbe5d4d9d26",
                    "type": "River",
                    "precipitation": {
                        "type": "Property",
                        "createdAt": "2021-05-04T06:45:32.674520Z",
                        "value": 2.2,
                        "observedAt": "2021-05-04T06:35:22.000Z",
                        "unitCode": "MMT"
                    },
                    "@context": [
                        "https://raw.githubusercontent.com/easy-global-market/ngsild-api-data-models/master/mlaas/jsonld-contexts/mlaas-precipitation-compound.jsonld"
                    ]
                }
            ]
        }
        input_data = input_data_notification['data'][0]['precipitation']['value']

        # reshaping input data into a 2D array
        input_data = np.array([input_data]).reshape(-1,1)

        ### CALLING BENTOML PREDICT HERE ###
        ###         HOW TO ???           ###
        ### What do you get back? numpy? ###
        ### More probably a list of list ###
        ### as received through HTTP     ###
        flow_prediction = [[2.3]]

        # Create NGSI-LD request to update Entity/Property
        # Here updating 'flow' Property of the Siagne Entity
        flow_prediction = round(float(np.array(flow_prediction).squeeze()), 2)
        timezone_France = pytz.timezone('Europe/Paris')
        predictedAt = timezone_France.localize(datetime.now().replace(microsecond=0)).isoformat()

        json_ = {
            '@context': AT_CONTEXT,
            'flow': [
                {
                    'type': 'Property',
                    'value': flow_prediction,
                    'unitCode': 'MQS',
                    'observedAt': predictedAt,
                    'conputedBy': {
                        'type': 'Relationship',
                        'object': MLMODEL_UUID
                    }
                }
            ]
        }

        URL_PATCH_FLOW = URL_ENTITIES+RIVER_SIAGNE_UUID+'/attrs'
        r = requests.post(URL_PATCH_FLOW, json=json_, headers=headers)

        # Finally, respond to the initial received request (notification)
        # with empty 200        
        response = make_response(
            '',
            200,
        )
        return response

    def setup_bento_service_api_routes(self):
        """
        Setup a route for each InferenceAPI object defined in bento_service
        """
        for api in self.bento_service.inference_apis:
            route_function = self.bento_service_api_func_wrapper(api)
            self.app.add_url_rule(
                rule="/{}".format(api.route),
                endpoint=api.name,
                view_func=route_function,
                methods=api.input_adapter.HTTP_METHODS,
            )

    def bento_service_api_func_wrapper(self, api: InferenceAPI):
        """
        Create api function for flask route, it wraps around user defined API
        callback and adapter class, and adds request logging and instrument metrics
        """

        def api_func():
            # handle_request may raise 4xx or 5xx exception.
            try:
                if request.headers.get(self.request_header_flag):
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

            return response

        def api_func_with_tracing():
            with get_tracer().span(
                service_name=f"BentoService.{self.bento_service.name}",
                span_name=f"InferenceAPI {api.name} HTTP route",
                request_headers=request.headers,
            ):
                return api_func()

        return api_func_with_tracing
