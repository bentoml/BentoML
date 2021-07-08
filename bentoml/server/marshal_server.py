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

from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer
from bentoml.marshal.marshal import MarshalService
from bentoml.server.instruments import setup_prometheus_multiproc_dir

marshal_logger = logging.getLogger("bentoml.marshal")


@inject
def gunicorn_marshal_server(
    default_workers: int = Provide[
        BentoMLContainer.config.bento_server.microbatch.workers
    ],
    default_timeout: int = Provide[BentoMLContainer.config.bento_server.timeout],
    default_outbound_workers: int = Provide[BentoMLContainer.api_server_workers],
    default_max_request_size: int = Provide[
        BentoMLContainer.config.bento_server.max_request_size
    ],
    default_port: int = Provide[BentoMLContainer.config.bento_server.port],
    default_mb_max_batch_size: int = Provide[
        BentoMLContainer.config.bento_server.microbatch.max_batch_size
    ],
    default_mb_max_latency: int = Provide[
        BentoMLContainer.config.bento_server.microbatch.max_latency
    ],
    default_loglevel: str = Provide[BentoMLContainer.config.bento_server.logging.level],
):
    from gunicorn.app.base import Application

    class GunicornMarshalServer(Application):  # pylint: disable=abstract-method
        @inject
        def __init__(
            self,
            bundle_path,
            outbound_host,
            outbound_port,
            workers: int = default_workers,
            timeout: int = default_timeout,
            outbound_workers: int = default_outbound_workers,
            max_request_size: int = default_max_request_size,
            port: int = default_port,
            mb_max_batch_size: int = default_mb_max_batch_size,
            mb_max_latency: int = default_mb_max_latency,
            prometheus_lock: Optional[multiprocessing.Lock] = None,
            loglevel: str = default_loglevel,
        ):
            self.bento_service_bundle_path = bundle_path

            self.port = port
            self.options = {
                "bind": "%s:%s" % ("0.0.0.0", self.port),
                "timeout": timeout,
                "limit_request_line": max_request_size,
                "loglevel": loglevel.upper(),
                "worker_class": "aiohttp.worker.GunicornWebWorker",
            }
            if workers:
                self.options['workers'] = workers
            self.prometheus_lock = prometheus_lock

            self.outbound_port = outbound_port
            self.outbound_host = outbound_host
            self.outbound_workers = outbound_workers
            self.mb_max_batch_size = mb_max_batch_size
            self.mb_max_latency = mb_max_latency

            super().__init__()

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
            server = MarshalService(
                self.bento_service_bundle_path,
                self.outbound_host,
                self.outbound_port,
                outbound_workers=self.outbound_workers,
                mb_max_batch_size=self.mb_max_batch_size,
                mb_max_latency=self.mb_max_latency,
            )
            return server.make_app()

        def run(self):
            setup_prometheus_multiproc_dir(self.prometheus_lock)
            marshal_logger.info("Running micro batch service on :%d", self.port)
            super().run()

    return GunicornMarshalServer
