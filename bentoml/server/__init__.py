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

from bentoml.configuration.containers import BentoMLConfiguration, BentoMLContainer
from bentoml.utils import reserve_free_port

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


def start_prod_server(
    saved_bundle_path: str,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    timeout: Optional[int] = None,
    enable_microbatch: Optional[bool] = None,
    enable_swagger: Optional[bool] = None,
    mb_max_batch_size: Optional[int] = None,
    mb_max_latency: Optional[int] = None,
    microbatch_workers: Optional[int] = None,
    config_file: Optional[str] = None,
):
    import psutil

    assert (
        psutil.POSIX
    ), "BentoML API Server production mode only supports POSIX platforms"

    config = BentoMLConfiguration(override_config_file=config_file)
    config.override(["api_server", "port"], port)
    config.override(["api_server", "workers"], workers)
    config.override(["api_server", "timeout"], timeout)
    config.override(["api_server", "enable_microbatch"], enable_microbatch)
    config.override(["api_server", "enable_swagger"], enable_swagger)
    config.override(["marshal_server", "max_batch_size"], mb_max_batch_size)
    config.override(["marshal_server", "max_latency"], mb_max_latency)
    config.override(["marshal_server", "workers"], microbatch_workers)

    if config.config['api_server'].get('enable_microbatch'):
        prometheus_lock = multiprocessing.Lock()
        with reserve_free_port() as api_server_port:
            pass

        model_server_job = multiprocessing.Process(
            target=_start_prod_server,
            kwargs=dict(
                saved_bundle_path=saved_bundle_path,
                port=api_server_port,
                config=config,
                prometheus_lock=prometheus_lock,
            ),
            daemon=True,
        )
        model_server_job.start()

        try:
            _start_prod_batching_server(
                saved_bundle_path=saved_bundle_path,
                config=config,
                api_server_port=api_server_port,
                prometheus_lock=prometheus_lock,
            )
        finally:
            model_server_job.terminate()
    else:
        _start_prod_server(saved_bundle_path=saved_bundle_path, config=config)


def _start_prod_server(
    saved_bundle_path: str,
    config: BentoMLConfiguration,
    port: Optional[int] = None,
    prometheus_lock: Optional[multiprocessing.Lock] = None,
):

    logger.info("Starting BentoML API server in production mode..")

    container = BentoMLContainer()
    container.config.from_dict(config.as_dict())

    from bentoml import server

    container.wire(packages=[server])

    if port is None:
        gunicorn_app = server.gunicorn_server.GunicornBentoServer(
            saved_bundle_path, prometheus_lock=prometheus_lock,
        )
    else:
        gunicorn_app = server.gunicorn_server.GunicornBentoServer(
            saved_bundle_path, port=port, prometheus_lock=prometheus_lock,
        )
    gunicorn_app.run()


def _start_prod_batching_server(
    saved_bundle_path: str,
    api_server_port: int,
    config: BentoMLConfiguration,
    prometheus_lock: Optional[multiprocessing.Lock] = None,
):

    logger.info("Starting BentoML Batching server in production mode..")

    container = BentoMLContainer()
    container.config.from_dict(config.as_dict())

    from bentoml import marshal, server

    container.wire(packages=[server, marshal])

    # avoid load model before gunicorn fork
    marshal_server = server.marshal_server.GunicornMarshalServer(
        bundle_path=saved_bundle_path,
        prometheus_lock=prometheus_lock,
        outbound_host="localhost",
        outbound_port=api_server_port,
    )
    marshal_server.run()
