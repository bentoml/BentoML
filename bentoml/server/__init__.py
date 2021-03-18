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
import sys
from typing import Optional

from bentoml.configuration.containers import BentoMLConfiguration, BentoMLContainer
from bentoml.utils import reserve_free_port

logger = logging.getLogger(__name__)


def start_dev_server(
    bundle_path: str,
    port: Optional[int] = None,
    enable_microbatch: Optional[bool] = None,
    mb_max_batch_size: Optional[int] = None,
    mb_max_latency: Optional[int] = None,
    run_with_ngrok: Optional[bool] = None,
    enable_swagger: Optional[bool] = None,
    config_file: Optional[str] = None,
):
    config = BentoMLConfiguration(override_config_file=config_file)
    config.override(["api_server", "port"], port)
    config.override(["api_server", "enable_microbatch"], enable_microbatch)
    config.override(["api_server", "enable_swagger"], enable_swagger)
    config.override(["marshal_server", "max_batch_size"], mb_max_batch_size)
    config.override(["marshal_server", "max_latency"], mb_max_latency)

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
                config=config,
            ),
            daemon=True,
        )
    model_server_proc.start()

    try:
        _start_dev_proxy(
            api_server_port=api_server_port,
            saved_bundle_path=bundle_path,
            config=config,
        )
    finally:
        model_server_proc.terminate()


def _start_dev_server(
    saved_bundle_path: str, api_server_port: int, config: BentoMLConfiguration,
):

    logger.info("Starting BentoML API server in development mode..")

    from bentoml.saved_bundle import load_from_dir

    bento_service = load_from_dir(saved_bundle_path)

    from bentoml.server.api_server import BentoAPIServer

    container = BentoMLContainer()
    container.config.from_dict(config.as_dict())
    container.wire(packages=[sys.modules[__name__]])

    api_server = BentoAPIServer(bento_service)
    api_server.start(port=api_server_port)


def _start_dev_proxy(
    saved_bundle_path: str, api_server_port: int, config: BentoMLConfiguration,
):

    logger.info("Starting BentoML API proxy in development mode..")

    from bentoml import marshal

    container = BentoMLContainer()
    container.config.from_dict(config.as_dict())
    container.wire(packages=[marshal])

    from bentoml.marshal.marshal import MarshalService

    marshal_server = MarshalService(
        saved_bundle_path, outbound_host="localhost", outbound_port=api_server_port,
    )

    marshal_server.fork_start_app()


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
        _start_prod_proxy(
            saved_bundle_path=saved_bundle_path,
            config=config,
            api_server_port=api_server_port,
            prometheus_lock=prometheus_lock,
        )
    finally:
        model_server_job.terminate()


def _start_prod_server(
    saved_bundle_path: str,
    config: BentoMLConfiguration,
    port: int,
    prometheus_lock: Optional[multiprocessing.Lock] = None,
):

    logger.info("Starting BentoML API server in production mode..")

    container = BentoMLContainer()
    container.config.from_dict(config.as_dict())

    container.wire(packages=[sys.modules[__name__]])

    from bentoml.server.gunicorn_server import GunicornBentoServer

    gunicorn_app = GunicornBentoServer(
        saved_bundle_path, port=port, prometheus_lock=prometheus_lock,
    )
    gunicorn_app.run()


def _start_prod_proxy(
    saved_bundle_path: str,
    api_server_port: int,
    config: BentoMLConfiguration,
    prometheus_lock: Optional[multiprocessing.Lock] = None,
):

    logger.info("Starting BentoML proxy in production mode..")

    container = BentoMLContainer()
    container.config.from_dict(config.as_dict())

    from bentoml import marshal
    from bentoml.server.marshal_server import GunicornMarshalServer

    container.wire(packages=[sys.modules[__name__], marshal])

    # avoid load model before gunicorn fork
    marshal_server = GunicornMarshalServer(
        bundle_path=saved_bundle_path,
        prometheus_lock=prometheus_lock,
        outbound_host="localhost",
        outbound_port=api_server_port,
    )
    marshal_server.run()
