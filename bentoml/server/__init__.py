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

from bentoml.utils import reserve_free_port

logger = logging.getLogger(__name__)


def start_dev_server(
    bundle_path: str,
    port: Optional[int] = None,
    mb_max_batch_size: Optional[int] = None,
    mb_max_latency: Optional[int] = None,
    run_with_ngrok: Optional[bool] = None,
    enable_swagger: Optional[bool] = None,
):
    if run_with_ngrok:
        from threading import Timer

        from bentoml.utils.flask_ngrok import start_ngrok

        thread = Timer(1, start_ngrok, args=(port,))
        thread.setDaemon(True)
        thread.start()

    with reserve_free_port() as api_server_port:
        # start server right after port released
        #  to reduce potential race

        model_server_proc = multiprocessing.Process(
            target=_start_dev_server,
            kwargs=dict(
                api_server_port=api_server_port,
                saved_bundle_path=bundle_path,
                enable_swagger=enable_swagger,
            ),
            daemon=True,
        )
    model_server_proc.start()

    try:
        _start_dev_proxy(
            port=port,
            api_server_port=api_server_port,
            saved_bundle_path=bundle_path,
            mb_max_batch_size=mb_max_batch_size,
            mb_max_latency=mb_max_latency,
        )
    finally:
        model_server_proc.terminate()


def _start_dev_server(
    saved_bundle_path: str, api_server_port: int, enable_swagger: bool,
):
    logger.info("Starting BentoML API server in development mode..")

    from bentoml.server.api_server import BentoAPIServer
    from bentoml.saved_bundle import load_from_dir

    bento_service = load_from_dir(saved_bundle_path)
    api_server = BentoAPIServer(bento_service, enable_swagger=enable_swagger)
    api_server.start(port=api_server_port)


def _start_dev_proxy(
    port: int,
    saved_bundle_path: str,
    api_server_port: int,
    mb_max_batch_size: int,
    mb_max_latency: int,
):
    logger.info("Starting BentoML API proxy in development mode..")

    from bentoml.marshal.marshal import MarshalService

    marshal_server = MarshalService(
        saved_bundle_path,
        outbound_host="localhost",
        outbound_port=api_server_port,
        mb_max_batch_size=mb_max_batch_size,
        mb_max_latency=mb_max_latency,
    )

    marshal_server.fork_start_app(port=port)


def start_prod_server(
    saved_bundle_path: str,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    timeout: Optional[int] = None,
    enable_swagger: Optional[bool] = None,
    mb_max_batch_size: Optional[int] = None,
    mb_max_latency: Optional[int] = None,
    microbatch_workers: Optional[int] = None,
):

    import psutil

    assert (
        psutil.POSIX
    ), "BentoML API Server production mode only supports POSIX platforms"

    prometheus_lock = multiprocessing.Lock()
    with reserve_free_port() as api_server_port:
        pass

    model_server_job = multiprocessing.Process(
        target=_start_prod_server,
        kwargs=dict(
            saved_bundle_path=saved_bundle_path,
            port=api_server_port,
            timeout=timeout,
            workers=workers,
            prometheus_lock=prometheus_lock,
            enable_swagger=enable_swagger,
        ),
        daemon=True,
    )
    model_server_job.start()

    try:
        _start_prod_proxy(
            saved_bundle_path=saved_bundle_path,
            port=port,
            api_server_port=api_server_port,
            workers=microbatch_workers,
            timeout=timeout,
            outbound_workers=workers,
            mb_max_batch_size=mb_max_batch_size,
            mb_max_latency=mb_max_latency,
            prometheus_lock=prometheus_lock,
        )
    finally:
        model_server_job.terminate()


def _start_prod_server(
    saved_bundle_path: str,
    port: int,
    timeout: int,
    workers: int,
    enable_swagger: bool,
    prometheus_lock: Optional[multiprocessing.Lock] = None,
):

    logger.info("Starting BentoML API server in production mode..")

    from bentoml.server.gunicorn_server import gunicorn_bento_server

    gunicorn_app = gunicorn_bento_server()(
        saved_bundle_path,
        port=port,
        timeout=timeout,
        workers=workers,
        prometheus_lock=prometheus_lock,
        enable_swagger=enable_swagger,
    )
    gunicorn_app.run()


def _start_prod_proxy(
    saved_bundle_path: str,
    port: int,
    api_server_port: int,
    workers: int,
    timeout: int,
    outbound_workers: int,
    mb_max_batch_size: int,
    mb_max_latency: int,
    prometheus_lock: Optional[multiprocessing.Lock] = None,
):

    logger.info("Starting BentoML proxy in production mode..")

    from bentoml.server.marshal_server import gunicorn_marshal_server

    # avoid load model before gunicorn fork
    marshal_server = gunicorn_marshal_server()(
        bundle_path=saved_bundle_path,
        prometheus_lock=prometheus_lock,
        port=port,
        workers=workers,
        timeout=timeout,
        outbound_host="localhost",
        outbound_port=api_server_port,
        outbound_workers=outbound_workers,
        mb_max_batch_size=mb_max_batch_size,
        mb_max_latency=mb_max_latency,
    )
    marshal_server.run()
