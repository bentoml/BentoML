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
from bentoml.marshal import MarshalService
from bentoml.utils.usage_stats import track_server
from bentoml.server.utils import setup_prometheus_multiproc_dir


marshal_logger = logging.getLogger("bentoml.marshal")


class GunicornMarshalServer(Application):  # pylint: disable=abstract-method
    def __init__(
        self,
        outbound_host,
        outbound_port,
        bundle_path,
        port=None,
        workers=1,
        timeout=None,
        prometheus_lock=None,
        outbound_workers=1,
    ):
        self.bento_service_bundle_path = bundle_path

        self.port = port or config("apiserver").getint("default_port")
        timeout = timeout or config("apiserver").getint("default_timeout")
        self.options = {
            "bind": "%s:%s" % ("0.0.0.0", self.port),
            "timeout": timeout,
            "loglevel": config("logging").get("LOGGING_LEVEL").upper(),
            "worker_class": "aiohttp.worker.GunicornWebWorker",
        }
        if workers:
            self.options['workers'] = workers
        self.prometheus_lock = prometheus_lock

        self.outbound_port = outbound_port
        self.outbound_host = outbound_host
        self.outbound_workers = outbound_workers

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
        )
        return server.make_app()

    def run(self):
        track_server('gunicorn-microbatch', {"number_of_workers": self.cfg.workers})
        setup_prometheus_multiproc_dir(self.prometheus_lock)
        super(GunicornMarshalServer, self).run()

    def async_run(self):
        """
        Start an micro batch server.
        """
        marshal_proc = multiprocessing.Process(target=self.run, daemon=True)
        marshal_proc.start()
        marshal_logger.info("Running micro batch service on :%d", self.port)
