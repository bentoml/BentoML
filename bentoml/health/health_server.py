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
from flask import Flask, Response, request
from simple_di import Provide, inject

from bentoml.configuration import get_debug_mode
from bentoml.configuration.containers import BentoMLContainer

CONTENT_TYPE_LATEST = str("text/plain; version=0.0.4; charset=utf-8")

feedback_logger = logging.getLogger("bentoml.feedback")
logger = logging.getLogger(__name__)


class BentoHealthServer:
    """
    HealthServer
    """

    @inject
    def __init__(
            self,
            port: int = Provide[BentoMLContainer.config.health.port],
            host: str = "localhost",
    ):
        app_name = "HealthServer"

        self.port = port
        self.host = host
        self.app = Flask(app_name, static_folder=None)

        self.setup_routes()

    def start(self):
        """
        Start a health server at the specific port on the instance or parameter.
        """
        # Bentoml api service is not thread safe.
        # Flask dev server enabled threaded by default, disable it.
        self.app.run(
            host=self.host,
            port=self.port,
            threaded=False,
            debug=get_debug_mode(),
            use_reloader=False,
        )

    @staticmethod
    def livez_view_func():
        """
        Health endpoint (liveness) for BentoML.
        Make sure it works with Kubernetes liveness probe
        """
        return Response(response="\n", status=200, mimetype="text/plain")

    def setup_routes(self):
        self.app.add_url_rule("/livez", "livez", self.livez_view_func)
