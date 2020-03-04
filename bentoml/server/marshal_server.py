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

from gunicorn.app.base import Application

from bentoml import config
from bentoml.bundler import load_bento_service_metadata
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
    _DEFAULT_MAX_BATCH_SIZE = config("marshal_server").getint("default_max_batch_size")

    def __init__(self, target_host, target_port, port=_DEFAULT_PORT):
        self.port = port
        self.marshal_app = MarshalService(target_host, target_port)

    def setup_routes_from_pb(self, bento_service_metadata_pb):
        for api_config in bento_service_metadata_pb.apis:
            if api_config.handler_type in HANDLER_TYPES_BATCH_MODE_SUPPORTED:
                handler_config = getattr(api_config, "handler_config", {})
                max_latency = (
                    handler_config["mb_max_latency"]
                    if "mb_max_latency" in handler_config
                    else self._DEFAULT_MAX_LATENCY
                )
                self.marshal_app.add_batch_handler(
                    api_config.name, max_latency, self._DEFAULT_MAX_BATCH_SIZE
                )
                marshal_logger.info("Micro batch enabled for API `%s`", api_config.name)

    def async_start(self):
        """
        Start an micro batch server at the specific port on the instance or parameter.
        """
        track_server('marshal')
        marshal_proc = multiprocessing.Process(
            target=self.marshal_app.fork_start_app,
            kwargs=dict(port=self.port),
            daemon=True,
        )
        # TODO: make sure child process dies when parent process is killed.
        marshal_proc.start()
        marshal_logger.info("Running micro batch service on :%d", self.port)


class GunicornMarshalServer(Application):  # pylint: disable=abstract-method
    def __init__(
        self,
        target_host,
        target_port,
        bundle_path,
        port=None,
        num_of_workers=1,
        timeout=None,
    ):
        self.bento_service_bundle_path = bundle_path

        self.target_port = target_port
        self.target_host = target_host
        self.port = port or config("apiserver").getint("default_port")
        timeout = timeout or config("apiserver").getint("default_timeout")
        self.options = {
            "bind": "%s:%s" % ("0.0.0.0", self.port),
            "timeout": timeout,
            "loglevel": config("logging").get("LOGGING_LEVEL").upper(),
            "worker_class": "aiohttp.worker.GunicornWebWorker",
        }
        if num_of_workers:
            self.options['workers'] = num_of_workers

        super(GunicornMarshalServer, self).__init__()

    def load_config(self):
        self.load_config_from_file("python:bentoml.server.gunicorn_config")

        # override config with self.options
        gunicorn_config = dict(
            [
                (key, value)
                for key, value in self.options.items()
                if key in self.cfg.settings and value is not None
            ]
        )
        for key, value in gunicorn_config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        server = MarshalServer(self.target_host, self.target_port, port=self.port)
        bento_service_metadata_pb = load_bento_service_metadata(
            self.bento_service_bundle_path
        )
        server.setup_routes_from_pb(bento_service_metadata_pb)
        return server.marshal_app.make_app()

    def run(self):
        track_server('gunicorn-microbatch', {"number_of_workers": self.cfg.workers})
        super(GunicornMarshalServer, self).run()

    def async_run(self):
        """
        Start an micro batch server.
        """
        marshal_proc = multiprocessing.Process(target=self.run, daemon=True,)
        marshal_proc.start()
        marshal_logger.info("Running micro batch service on :%d", self.port)
