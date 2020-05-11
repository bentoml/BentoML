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

import os
import shutil
import multiprocessing
import logging

from bentoml import config

logger = logging.getLogger(__name__)


def get_gunicorn_num_of_workers():
    if config("apiserver").getint("default_gunicorn_workers_count") > 0:
        num_of_workers = config("apiserver").getint("default_gunicorn_workers_count")
        logger.info(
            "get_gunicorn_num_of_workers: %d, loaded from config", num_of_workers
        )
    else:
        num_of_workers = (multiprocessing.cpu_count() // 2) + 1
        logger.info(
            "get_gunicorn_num_of_workers: %d, calculated by cpu count", num_of_workers
        )

    return num_of_workers


def setup_prometheus_multiproc_dir(lock: multiprocessing.Lock = None):
    """
    Set up prometheus_multiproc_dir for prometheus to work in multiprocess mode,
    which is required when working with Gunicorn server

    Warning: for this to work, prometheus_client library must be imported after
    this function is called. It relies on the os.environ['prometheus_multiproc_dir']
    to properly setup for multiprocess mode
    """
    if lock is not None:
        lock.acquire()

    try:
        prometheus_multiproc_dir = config('instrument').get('prometheus_multiproc_dir')
        logger.debug(
            "Setting up prometheus_multiproc_dir: %s", prometheus_multiproc_dir
        )
        # Wipe prometheus metrics directory between runs
        # https://github.com/prometheus/client_python#multiprocess-mode-gunicorn
        # Ignore errors so it does not fail when directory does not exist
        shutil.rmtree(prometheus_multiproc_dir, ignore_errors=True)
        os.makedirs(prometheus_multiproc_dir, exist_ok=True)

        os.environ['prometheus_multiproc_dir'] = prometheus_multiproc_dir
    finally:
        if lock is not None:
            lock.release()
