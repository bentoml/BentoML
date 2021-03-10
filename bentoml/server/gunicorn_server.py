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

import psutil
from dependency_injector.wiring import Provide as P
from dependency_injector.wiring import inject
from flask import Response

from bentoml.configuration.containers import BentoMLContainer as C
from bentoml.saved_bundle import load_from_dir
from bentoml.server.api_server import BentoAPIServer
from bentoml.server.instruments import setup_prometheus_multiproc_dir

logger = logging.getLogger(__name__)


class GunicornBentoAPIServer(BentoAPIServer):
    def metrics_view_func(self):
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            generate_latest,
            multiprocess,
        )

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)


if psutil.POSIX:
    from gunicorn.app.base import Application

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
            port: int = P[C.config.api_server.port],
            timeout: int = P[C.config.api_server.timeout],
            workers: int = P[C.api_server_workers],
            prometheus_lock: Optional[multiprocessing.Lock] = None,
            enable_swagger: bool = P[C.config.api_server.enable_swagger],
            max_request_size: int = P[C.config.api_server.max_request_size],
            loglevel=P[C.config.logging.level],
        ):
            self.bento_service_bundle_path = bundle_path

            if bind is None:
                self.bind = f"0.0.0.0:{port}"
            else:
                self.bind = bind

            self.options = {
                "bind": self.bind,
                "timeout": timeout,
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


else:

    class GunicornBentoServer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "GunicornBentoServer is not supported in non-POSIX environments."
            )
