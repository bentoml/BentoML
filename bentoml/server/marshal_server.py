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

from __future__ import absolute_import, division, print_function

import logging
import multiprocessing

from bentoml import config
from bentoml.marshal import MarshalService
from bentoml.handlers import HANDLER_TYPES_BATCH_MODE_SUPPORTED
from bentoml.utils.usage_stats import track_server


marshal_logger = logging.getLogger("bentoml.marshal")


class MarshalServer:
    """
    MarshalServer creates a reverse proxy server in front of actual API server,
    implementing the micro batching feature.
    Requests in a short period(mb_max_latency) are collected and sent to API server,
    merged into a single request.
    """

    _DEFAULT_PORT = config("apiserver").getint("default_port")
    _DEFAULT_MAX_LATENCY = config("marshal_server").getint("default_max_latency")

    def __init__(self, target_host, target_port, port=_DEFAULT_PORT):
        self.port = port
        self.marshal_app = MarshalService(target_host, target_port)

    def setup_routes_from_pb(self, bento_service_metadata_pb):
        for api_config in bento_service_metadata_pb.apis:
            if api_config.handler_type in HANDLER_TYPES_BATCH_MODE_SUPPORTED:
                handler_config = getattr(api_config, "handler_config", {})
                max_latency = (handler_config["mb_max_latency"]
                               if "mb_max_latency" in handler_config
                               else self._DEFAULT_MAX_LATENCY)
                self.marshal_app.add_batch_handler(api_config.name, max_latency)
                marshal_logger.info("Micro batch enabled for API `%s`", api_config.name)

    def async_start(self):
        """
        Start an micro batch server at the specific port on the instance or parameter.
        """
        track_server('marshal')
        marshal_proc = multiprocessing.Process(
            target=self.marshal_app.fork_start_app,
            kwargs=dict(port=self.port),
            daemon=True)
        # TODO: make sure child process dies when parent process is killed.
        marshal_proc.start()
        marshal_logger.info("Running micro batch service on :%d", self.port)
