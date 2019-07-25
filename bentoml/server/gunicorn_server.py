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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

from gunicorn.app.base import BaseApplication
from gunicorn.six import iteritems

from bentoml import config
from bentoml.archive import load
from bentoml.server import BentoAPIServer

conf = config["apiserver"]


def get_gunicorn_worker_count():
    """
    Generate an recommend gunicorn worker process count

    Gunicorn's documentation recommand 2 times of cpu cores + 1.
    For ml model serving, it might consumer more computing resources, therefore
    we recommend half of the number of cpu cores + 1
    """

    return (multiprocessing.cpu_count() // 2) + 1


class GunicornBentoServer(BaseApplication):  # pylint: disable=abstract-method
    """
    A custom Gunicorn application.

    Usage::

        >>> from bentoml.server.gunicorn_server import GunicornBentoServer
        >>>
        >>> gunicorn_app = GunicornBentoServer(bento_archive_path, port=5000)
        >>> gunicorn_app.run()

    :param app: a Flask app, flask.Flask.app
    :param port: the port you want to run gunicorn server on
    :param workers: number of worker processes
    """

    _DEFAULT_PORT = conf.getint("default_port")
    _DEFAULT_TIMEOUT = conf.getint("default_timeout")

    def __init__(
        self, bento_archive_path, port=None, num_of_workers=None, timeout=None
    ):
        self.bento_archive_path = bento_archive_path
        self.port = port or self._DEFAULT_PORT
        self.num_of_workers = num_of_workers or get_gunicorn_worker_count()
        self.timeout = timeout or self._DEFAULT_TIMEOUT

        self.options = {
            "workers": self.num_of_workers,
            "bind": "%s:%s" % ("0.0.0.0", self.port),
            "timeout": self.timeout,
        }
        super(GunicornBentoServer, self).__init__()

    def load_config(self):
        gunicorn_config = dict(
            [
                (key, value)
                for key, value in iteritems(self.options)
                if key in self.cfg.settings and value is not None
            ]
        )
        for key, value in iteritems(gunicorn_config):
            self.cfg.set(key.lower(), value)

    def load(self):
        bento_service = load(self.bento_archive_path)
        api_server = BentoAPIServer(bento_service, port=self.port)
        return api_server.app
