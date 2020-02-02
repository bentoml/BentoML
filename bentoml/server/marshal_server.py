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


CONTENT_TYPE_LATEST = str("text/plain; version=0.0.4; charset=utf-8")

marshal_logger = logging.getLogger("bentoml.marshal")
logger = logging.getLogger(__name__)


class MarshalServer:
    """
    BentoAPIServer creates a REST API server based on APIs defined with a BentoService
    via BentoService#get_service_apis call. Each BentoServiceAPI will become one
    endpoint exposed on the REST server, and the RequestHandler defined on each
    BentoServiceAPI object will be used to handle Request object before feeding the
    request data into a Service API function
    """

    _DEFAULT_PORT = config("apiserver").getint("default_port")
    _DEFAULT_MAX_LATENCY = config("marshal_server").getint("default_max_latency")

    def __init__(self, bento_service, target_host, target_port,
                 port=_DEFAULT_PORT, app_name=None):
        app_name = bento_service.name if app_name is None else app_name

        self.port = port
        self.bento_service = bento_service

        self.marshal_app = MarshalService(target_host, target_port)
        self.setup_routes()

    def setup_routes(self):
        for api in self.bento_service.get_service_apis():
            if getattr(api.handler, "micro_batch", False):
                max_latency = getattr(api.handler, "mb_max_latency",
                                      self._DEFAULT_MAX_LATENCY)
                self.marshal_app.add_batch_handler(api.name, max_latency)

    def async_start(self):
        """
        Start an REST server at the specific port on the instance or parameter.
        """
        # track_server('marshal')
        marshal_proc = multiprocessing.Process(
            target=self.marshal_app.fork_start_app,
            kwargs=dict(port=self.port),
            daemon=True)
        # TODO: make sure child process dies when parent process is killed.
        marshal_proc.start()
