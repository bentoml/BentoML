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
    saved_bundle_path: str,
    port: int,
    enable_microbatch: bool,
    mb_max_batch_size: int,
    mb_max_latency: int,
    run_with_ngrok: bool,
    enable_swagger: bool,
):
    logger.info("Starting BentoML API server in development mode..")

    import multiprocessing

    from bentoml import load
    from bentoml.server.api_server import BentoAPIServer
    from bentoml.utils import reserve_free_port

    if run_with_ngrok:
        from threading import Timer

        from bentoml.utils.flask_ngrok import start_ngrok

        thread = Timer(1, start_ngrok, args=(port,))
        thread.setDaemon(True)
        thread.start()

    if enable_microbatch:
        with reserve_free_port() as api_server_port:
            # start server right after port released
            #  to reduce potential race

            marshal_proc = multiprocessing.Process(
                target=start_dev_batching_server,
                kwargs=dict(
                    api_server_port=api_server_port,
                    saved_bundle_path=saved_bundle_path,
                    port=port,
                    mb_max_latency=mb_max_latency,
                    mb_max_batch_size=mb_max_batch_size,
                ),
                daemon=True,
            )
        marshal_proc.start()

        bento_service = load(saved_bundle_path)
        api_server = BentoAPIServer(
            bento_service, port=api_server_port, enable_swagger=enable_swagger
        )
        api_server.start()
    else:
        bento_service = load(saved_bundle_path)
        api_server = BentoAPIServer(
            bento_service, port=port, enable_swagger=enable_swagger
        )
        api_server.start()


def start_dev_batching_server(
    saved_bundle_path: str,
    port: int,
    api_server_port: int,
    mb_max_batch_size: int,
    mb_max_latency: int,
):

    from bentoml.marshal.marshal import MarshalService

    marshal_server = MarshalService(
        saved_bundle_path,
        outbound_host="localhost",
        outbound_port=api_server_port,
        outbound_workers=1,
        mb_max_batch_size=mb_max_batch_size,
        mb_max_latency=mb_max_latency,
    )
    logger.info("Running micro batch service on :%d", port)
    marshal_server.fork_start_app(port=port)


def start_prod_server(
    saved_bundle_path: str,
    port: int,
    timeout: int,
    workers: int,
    enable_microbatch: bool,
    mb_max_batch_size: int,
    mb_max_latency: int,
    microbatch_workers: int,
    enable_swagger: bool,
):
    logger.info("Starting BentoML API server in production mode..")

    import multiprocessing

    import psutil

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
                mb_max_batch_size=mb_max_batch_size,
                mb_max_latency=mb_max_latency,
            )

            gunicorn_app = GunicornBentoServer(
                saved_bundle_path,
                api_server_port,
                workers,
                timeout,
                prometheus_lock,
                enable_swagger,
            )
        marshal_server.async_run()
        gunicorn_app.run()
    else:
        gunicorn_app = GunicornBentoServer(
            saved_bundle_path, port, workers, timeout, enable_swagger=enable_swagger
        )
        gunicorn_app.run()
