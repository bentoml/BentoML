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
from bentoml.utils import ProtoMessageToDict, reserve_free_port, resolve_bundle_path

logger = logging.getLogger(__name__)


def start_prod_server(saved_bundle_path: str, config: BentoMLConfiguration):
    import psutil

    assert (
        psutil.POSIX
    ), "BentoML API Server production mode only supports POSIX platforms"

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
