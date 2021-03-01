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

from dependency_injector.wiring import Provide, inject

from bentoml.configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)


@inject
def start_dev_server(
    saved_bundle_path: str,
    port: int = Provide[BentoMLContainer.config.api_server.port],
    enable_microbatch: bool = Provide[
        BentoMLContainer.config.api_server.enable_microbatch
    ],
    mb_max_batch_size: int = Provide[
        BentoMLContainer.config.marshal_server.max_batch_size
    ],
    mb_max_latency: int = Provide[BentoMLContainer.config.marshal_server.max_latency],
    run_with_ngrok: bool = Provide[BentoMLContainer.config.api_server.run_with_ngrok],
    enable_swagger: bool = Provide[BentoMLContainer.config.api_server.enable_swagger],
):
    logger.info("Starting BentoML API server in development mode..")

    from bentoml.saved_bundle import load_from_dir
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

        bento_service = load_from_dir(saved_bundle_path)
        api_server = BentoAPIServer(bento_service, enable_swagger=enable_swagger)
        api_server.start(port=api_server_port)
    else:
        bento_service = load_from_dir(saved_bundle_path)
        api_server = BentoAPIServer(bento_service, enable_swagger=enable_swagger)
        api_server.start(port=port)


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


@inject
def start_prod_server(
    saved_bundle_path: str,
    port: int = Provide[BentoMLContainer.config.api_server.port],
    timeout: int = Provide[BentoMLContainer.config.api_server.timeout],
    workers: int = Provide[BentoMLContainer.api_server_workers],
    enable_swagger: bool = Provide[BentoMLContainer.config.api_server.enable_swagger],
    prometheus_lock: Optional[multiprocessing.Lock] = None,
):
    logger.info("Starting BentoML API server in production mode..")

    import psutil

    assert (
        psutil.POSIX
    ), "BentoML API Server production mode only supports POSIX platforms"

    from bentoml.server.gunicorn_server import GunicornBentoServer

    gunicorn_app = GunicornBentoServer(
        saved_bundle_path,
        bind=f"0.0.0.0:{port}",
        workers=workers,
        timeout=timeout,
        enable_swagger=enable_swagger,
        prometheus_lock=prometheus_lock,
    )
    gunicorn_app.run()


@inject
def start_prod_batching_server(
    saved_bundle_path: str,
    api_server_port: int,
    api_server_workers: int = Provide[BentoMLContainer.api_server_workers],
    port: int = Provide[BentoMLContainer.config.api_server.port],
    workers: int = Provide[BentoMLContainer.config.marshal_server.workers],
    mb_max_batch_size: int = Provide[
        BentoMLContainer.config.marshal_server.max_batch_size
    ],
    mb_max_latency: int = Provide[BentoMLContainer.config.marshal_server.max_latency],
    prometheus_lock: Optional[multiprocessing.Lock] = None,
):

    import psutil

    assert (
        psutil.POSIX
    ), "BentoML Batching Server production mode only supports POSIX platforms"

    from bentoml.server.marshal_server import GunicornMarshalServer

    # avoid load model before gunicorn fork
    marshal_server = GunicornMarshalServer(
        bundle_path=saved_bundle_path,
        port=port,
        workers=workers,
        prometheus_lock=prometheus_lock,
        outbound_host="localhost",
        outbound_port=api_server_port,
        outbound_workers=api_server_workers,
        mb_max_batch_size=mb_max_batch_size,
        mb_max_latency=mb_max_latency,
    )
    marshal_server.run()
