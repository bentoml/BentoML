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

marshal_logger = logging.getLogger("bentoml.marshal")


if TYPE_CHECKING:  # make type checkers happy
    Lock = multiprocessing.synchronize.Lock
    from bentoml.marshal.marshal import MarshalApp


class GunicornMarshalServer(Application):  # pylint: disable=abstract-method
    @inject(squeeze_none=True)
    def __init__(
        self,
        *,
        app: "MarshalApp" = Provide[BentoMLContainer.proxy_app],
        workers: int = Provide[BentoMLContainer.config.bento_server.microbatch.workers],
        timeout: int = Provide[BentoMLContainer.config.bento_server.timeout],
        max_request_size: int = Provide[
            BentoMLContainer.config.bento_server.max_request_size
        ],
        host: str = Provide[BentoMLContainer.service_host],
        port: int = Provide[BentoMLContainer.service_port],
        prometheus_lock: Optional["Lock"] = Provide[BentoMLContainer.prometheus_lock],
        loglevel: str = Provide[BentoMLContainer.config.bento_server.logging.level],
    ):
        self.app = app
        self.port = port
        self.options = {
            "bind": "%s:%s" % (host, port),
            "timeout": timeout,
            "limit_request_line": max_request_size,
            "loglevel": loglevel.upper(),
            "worker_class": "aiohttp.worker.GunicornWebWorker",
        }
        if workers:
            self.options['workers'] = workers
        self.prometheus_lock = prometheus_lock
        super().__init__()

    def load_config(self):
        self.load_config_from_file("python:bentoml.server.gunicorn_config")

        # override config with self.options
        assert self.cfg, "gunicorn config must be loaded"
        for k, v in self.options.items():
            if k.lower() in self.cfg.settings and v is not None:
                self.cfg.set(k.lower(), v)

    def load(self):
        return self.app.make_aiohttp_app()

    def run(self):
        setup_prometheus_multiproc_dir(self.prometheus_lock)
        marshal_logger.info("Running micro batch service on :%d", self.port)
        super().run()
