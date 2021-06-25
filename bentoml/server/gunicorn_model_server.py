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
from typing import Optional, TYPE_CHECKING

from gunicorn.app.base import Application
from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer
from bentoml.server.instruments import setup_prometheus_multiproc_dir

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # make type checkers happy
    Lock = multiprocessing.synchronize.Lock
    from bentoml.server.model_app import ModelApp


class GunicornModelServer(Application):  # pylint: disable=abstract-method
    """
    A custom Gunicorn application.

    Usage::

        >>> from bentoml.server.gunicorn_model_server import GunicornModelServer
        >>>
        >>> gunicorn_app = GunicornModelServer(host="127.0.0.1", port=5000)
        >>> gunicorn_app.run()

    :param bind: Specify a server socket to bind. Server sockets can be any of
        $(HOST), $(HOST):$(PORT), fd://$(FD), or unix:$(PATH).
    :param workers: number of worker processes
    :param timeout: request timeout config
    """

    @inject
    def __init__(
        self,
        *,
        app: "ModelApp" = Provide[BentoMLContainer.model_app],
        host: str = Provide[BentoMLContainer.forward_host],
        port: int = Provide[BentoMLContainer.forward_port],
        timeout: int = Provide[BentoMLContainer.config.bento_server.timeout],
        workers: int = Provide[BentoMLContainer.api_server_workers],
        prometheus_lock: Optional["Lock"] = Provide[BentoMLContainer.prometheus_lock],
        max_request_size: int = Provide[
            BentoMLContainer.config.bento_server.max_request_size
        ],
        loglevel: str = Provide[BentoMLContainer.config.bento_server.logging.level],
    ):
        bind = f"{host}:{port}"

        self.options = {
            "bind": bind,
            "timeout": timeout,  # TODO
            "limit_request_line": max_request_size,
            "loglevel": loglevel.upper(),
        }
        if workers:
            self.options['workers'] = workers
        self.prometheus_lock = prometheus_lock
        self.app = app

        super().__init__()

    def load_config(self):
        self.load_config_from_file("python:bentoml.server.gunicorn_config")

        # override config with self.options
        assert self.cfg, "gunicorn config must be loaded"
        for k, v in self.options.items():
            if k.lower() in self.cfg.settings and v is not None:
                self.cfg.set(k.lower(), v)

    def load(self):
        return self.app.make_flask_app()

    def run(self):
        logger.info("Starting BentoML API server in production mode..")
        setup_prometheus_multiproc_dir(self.prometheus_lock)
        super().run()
