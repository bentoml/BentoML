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
import multiprocessing
from typing import Optional

from flask import Response
from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer
from bentoml.saved_bundle import load_from_dir
from bentoml.server.api_server import BentoAPIServer
from bentoml.server.instruments import setup_prometheus_multiproc_dir

logger = logging.getLogger(__name__)


@inject
def gunicorn_bento_server(
    default_port: int = Provide[BentoMLContainer.config.bento_server.port],
    default_timeout: int = Provide[BentoMLContainer.config.bento_server.timeout],
    default_workers: int = Provide[BentoMLContainer.api_server_workers],
    default_enable_swagger: bool = Provide[
        BentoMLContainer.config.bento_server.swagger.enabled
    ],
    default_max_request_size: int = Provide[
        BentoMLContainer.config.bento_server.max_request_size
    ],
    default_loglevel=Provide[BentoMLContainer.config.bento_server.logging.level],
):
    from gunicorn.app.base import Application

    class GunicornBentoAPIServer(BentoAPIServer):
        def metrics_view_func(self):
            from prometheus_client import (
                CONTENT_TYPE_LATEST,
                CollectorRegistry,
                generate_latest,
                multiprocess,
            )

            registry = CollectorRegistry()
            # NOTE: enable mb metrics to be parsed.
            multiprocess.MultiProcessCollector(registry)
            return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST,)

    class GunicornBentoServer(Application):  # pylint: disable=abstract-method
        """
        A custom Gunicorn application.

        Usage::

            >>> from bentoml.server.gunicorn_server import GunicornBentoServer
            >>>
            >>> gunicorn_app = GunicornBentoServer(saved_bundle_path,\
            >>>     bind="127.0.0.1:5000")
            >>> gunicorn_app.run()

        :param bundle_path: path to the saved BentoService bundle
        :param bind: Specify a server socket to bind. Server sockets can be any of
            $(HOST), $(HOST):$(PORT), fd://$(FD), or unix:$(PATH).
        :param workers: number of worker processes
        :param timeout: request timeout config
        """

        @inject
        def __init__(
            self,
            bundle_path,
            bind: str = None,
            port: int = default_port,
            timeout: int = default_timeout,
            workers: int = default_workers,
            prometheus_lock: Optional[multiprocessing.Lock] = None,
            enable_swagger: bool = default_enable_swagger,
            max_request_size: int = default_max_request_size,
            loglevel: str = default_loglevel,
        ):
            self.bento_service_bundle_path = bundle_path

            if bind is None:
                self.bind = f"0.0.0.0:{port}"
            else:
                self.bind = bind

            self.options = {
                "bind": self.bind,
                "timeout": timeout,  # TODO
                "limit_request_line": max_request_size,
                "loglevel": loglevel.upper(),
            }
            if workers:
                self.options['workers'] = workers
            self.prometheus_lock = prometheus_lock
            self.enable_swagger = enable_swagger

            super(GunicornBentoServer, self).__init__()

        def load_config(self):
            self.load_config_from_file("python:bentoml.server.gunicorn_config")

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
            bento_service = load_from_dir(self.bento_service_bundle_path)
            api_server = GunicornBentoAPIServer(
                bento_service, enable_swagger=self.enable_swagger
            )
            return api_server.app

        def run(self):
            setup_prometheus_multiproc_dir(self.prometheus_lock)
            super(GunicornBentoServer, self).run()

    return GunicornBentoServer
