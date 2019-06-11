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


def get_gunicorn_worker_count():
    """
    Generate an recommend gunicorn worker process count

    Gunicorn's documentation recommand 2 times of cpu cores + 1.
    For ml model serving, it might consumer more computing resources, therefore
    we recommend half of the number of cpu cores + 1
    """

    return (multiprocessing.cpu_count() // 2) + 1


class GunicornApplication(BaseApplication):  # pylint: disable=abstract-method
    """
    A custom Gunicorn application.

    Usage::

        >>> from bentoml.server.gunicorn_server import GunicornApplication
        >>>
        >>> gunicorn_app = GunicornApplication(app, 5000, 2)
        >>> gunicorn_app.run()

    :param app: a Flask app, flask.Flask.app
    :param port: the port you want to run gunicorn server on
    :param workers: number of worker processes
    """

    def __init__(self, app, port, workers, timeout):
        self.options = {
            "workers": workers,
            "bind": "%s:%s" % ("0.0.0.0", port),
            "timeout": timeout,
        }
        self.application = app
        super(GunicornApplication, self).__init__()

    def load_config(self):
        config = dict(
            [
                (key, value)
                for key, value in iteritems(self.options)
                if key in self.cfg.settings and value is not None
            ]
        )
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application
