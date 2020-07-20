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
from contextlib import contextmanager

from bentoml import config


logger = logging.getLogger(__name__)
ZIPKIN_API_URL = config("tracing").get("zipkin_api_url")


@contextmanager
def trace(*args, **kwargs):
    if ZIPKIN_API_URL:
        from bentoml.server.trace import trace as _trace

        return _trace(ZIPKIN_API_URL, *args, **kwargs)
    else:
        yield


@contextmanager
def async_trace(*args, **kwargs):
    if ZIPKIN_API_URL:
        from bentoml.server.trace import async_trace as _async_trace

        return _async_trace(ZIPKIN_API_URL, *args, **kwargs)
    else:
        yield


def start_dev_server(
    saved_bundle_path: str, port: int, enable_microbatch: bool, run_with_ngrok: bool
):
    logger.info("Starting BentoML API server in development mode..")

    from bentoml import load
    from bentoml.server.api_server import BentoAPIServer
    from bentoml.marshal.marshal import MarshalService
    from bentoml.utils import reserve_free_port

    bento_service = load(saved_bundle_path)

    if run_with_ngrok:
        from bentoml.utils.flask_ngrok import start_ngrok
        from threading import Timer

        thread = Timer(1, start_ngrok, args=(port,))
        thread.setDaemon(True)
        thread.start()

    if enable_microbatch:
        with reserve_free_port() as api_server_port:
            # start server right after port released
            #  to reduce potential race
            marshal_server = MarshalService(
                saved_bundle_path,
                outbound_host="localhost",
                outbound_port=api_server_port,
                outbound_workers=1,
            )
            api_server = BentoAPIServer(bento_service, port=api_server_port)
        marshal_server.async_start(port=port)
        api_server.start()
    else:
        api_server = BentoAPIServer(bento_service, port=port)
        api_server.start()


def start_prod_server(
    saved_bundle_path: str,
    port: int,
    timeout: int,
    workers: int,
    enable_microbatch: bool,
    microbatch_workers: int,
):
    logger.info("Starting BentoML API server in production mode..")

    import psutil
    import multiprocessing

    assert (
        psutil.POSIX
    ), "BentoML API Server production mode only supports POSIX platforms"

    from bentoml.server.gunicorn_server import GunicornBentoServer
    from bentoml.server.marshal_server import GunicornMarshalServer
    from bentoml.server.utils import get_gunicorn_num_of_workers
    from bentoml.utils import reserve_free_port

    if workers is None:
        workers = get_gunicorn_num_of_workers()

    if enable_microbatch:
        prometheus_lock = multiprocessing.Lock()
        # avoid load model before gunicorn fork
        with reserve_free_port() as api_server_port:
            marshal_server = GunicornMarshalServer(
                bundle_path=saved_bundle_path,
                port=port,
                workers=microbatch_workers,
                prometheus_lock=prometheus_lock,
                outbound_host="localhost",
                outbound_port=api_server_port,
                outbound_workers=workers,
            )

            gunicorn_app = GunicornBentoServer(
                saved_bundle_path, api_server_port, workers, timeout, prometheus_lock,
            )
        marshal_server.async_run()
        gunicorn_app.run()
    else:
        gunicorn_app = GunicornBentoServer(saved_bundle_path, port, workers, timeout)
        gunicorn_app.run()
