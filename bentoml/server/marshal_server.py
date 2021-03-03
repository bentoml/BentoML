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

from bentoml.configuration.containers import BentoMLContainer as C
from bentoml.marshal.marshal import MarshalService
from bentoml.server.instruments import setup_prometheus_multiproc_dir

marshal_logger = logging.getLogger("bentoml.marshal")


if psutil.POSIX:
    from gunicorn.app.base import Application

    class GunicornMarshalServer(Application):  # pylint: disable=abstract-method
        @inject
        def __init__(
            self,
            bundle_path,
            outbound_host,
            outbound_port,
            workers: int = P[C.config.marshal_server.workers],
            timeout: int = P[C.config.api_server.timeout],
            outbound_workers: int = P[C.api_server_workers],
            max_request_size: int = P[C.config.api_server.max_request_size],
            port: int = P[C.config.api_server.port],
            mb_max_batch_size: int = P[C.config.marshal_server.max_batch_size],
            mb_max_latency: int = P[C.config.marshal_server.max_latency],
            prometheus_lock: Optional[multiprocessing.Lock] = None,
            loglevel=P[C.config.logging.level],
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
            super(GunicornMarshalServer, self).run()


else:

    class GunicornMarshalServer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "GunicornMarshalServer is not supported in non-POSIX environments."
            )
