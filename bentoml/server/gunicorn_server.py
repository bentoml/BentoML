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
from flask import Response
from gunicorn.app.base import Application

from bentoml import config
from bentoml.saved_bundle import load
from bentoml.server.api_server import BentoAPIServer
from bentoml.server.instruments import setup_prometheus_multiproc_dir

logger = logging.getLogger(__name__)


class GunicornBentoAPIServer(BentoAPIServer):
    def metrics_view_func(self):
        from prometheus_client import (
            multiprocess,
            CollectorRegistry,
            generate_latest,
            CONTENT_TYPE_LATEST,
        )

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)


class GunicornBentoServer(Application):  # pylint: disable=abstract-method
    """
    A custom Gunicorn application.

    Usage::

        >>> from bentoml.server.gunicorn_server import GunicornBentoServer
        >>>
        >>> gunicorn_app = GunicornBentoServer(saved_bundle_path, port=5000)
        >>> gunicorn_app.run()

    :param bundle_path: path to the saved BentoService bundle
    :param port: the port you want to run gunicorn server on
    :param workers: number of worker processes
    :param timeout: request timeout config
    """

    def __init__(
        self, bundle_path, port=None, workers=None, timeout=None, prometheus_lock=None,
    ):
        self.bento_service_bundle_path = bundle_path

        self.port = port or config("apiserver").getint("default_port")
        timeout = timeout or config("apiserver").getint("default_timeout")
        self.options = {
            "bind": "%s:%s" % ("0.0.0.0", self.port),
            "timeout": timeout,
            "loglevel": config("logging").get("LOGGING_LEVEL").upper(),
        }
        if workers:
            self.options['workers'] = workers
        self.prometheus_lock = prometheus_lock

        super(GunicornBentoServer, self).__init__()

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
        bento_service = load(self.bento_service_bundle_path)
        api_server = GunicornBentoAPIServer(bento_service, port=self.port)
        return api_server.app

    def run(self):
        setup_prometheus_multiproc_dir(self.prometheus_lock)
        super(GunicornBentoServer, self).run()
