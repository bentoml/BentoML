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
from typing import NoReturn, Optional

from simple_di import skip, sync_container

from bentoml.configuration.containers import BentoMLContainer


logger = logging.getLogger(__name__)


def start_dev_server(
    bundle_path: str,
    port: Optional[int] = None,
    mb_max_batch_size: Optional[int] = None,
    mb_max_latency: Optional[int] = None,
    run_with_ngrok: Optional[bool] = None,
    enable_swagger: Optional[bool] = None,
    timeout: Optional[int] = None,
):
    BentoMLContainer.bundle_path.set(bundle_path)

    bento_server = BentoMLContainer.config.bento_server
    bento_server.port.set(port or skip)
    bento_server.timeout.set(timeout or skip)
    bento_server.microbatch.timeout.set(timeout or skip)
    bento_server.swagger.enabled.set(enable_swagger or skip)
    bento_server.microbatch.max_batch_size.set(mb_max_batch_size or skip)
    bento_server.microbatch.max_latency.set(mb_max_latency or skip)

    BentoMLContainer.prometheus_lock.get()  # generate lock before fork
    BentoMLContainer.forward_port.get()  # generate port before fork

    if run_with_ngrok:
        from threading import Timer

        from bentoml.utils.flask_ngrok import start_ngrok

        thread = Timer(1, start_ngrok, args=(port,))
        thread.setDaemon(True)
        thread.start()

    import multiprocessing

    model_server_proc = multiprocessing.Process(
        target=_start_dev_server, args=(BentoMLContainer,), daemon=True,
    )
    model_server_proc.start()

    try:
        _start_dev_proxy(BentoMLContainer)
    finally:
        model_server_proc.terminate()


def start_prod_server(
    bundle_path: str,
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

    BentoMLContainer.bundle_path.set(bundle_path)

    bento_server = BentoMLContainer.config.bento_server
    bento_server.port.set(port or skip)
    bento_server.timeout.set(timeout or skip)
    bento_server.microbatch.timeout.set(timeout or skip)
    bento_server.workers.set(workers or skip)
    bento_server.swagger.enabled.set(enable_swagger or skip)
    bento_server.microbatch.workers.set(microbatch_workers or skip)
    bento_server.microbatch.max_batch_size.set(mb_max_batch_size or skip)
    bento_server.microbatch.max_latency.set(mb_max_latency or skip)

    BentoMLContainer.prometheus_lock.get()  # generate lock before fork
    BentoMLContainer.forward_port.get()  # generate port before fork

    import multiprocessing

    model_server_job = multiprocessing.Process(
        target=_start_prod_server, args=(BentoMLContainer,), daemon=True
    )
    model_server_job.start()

    try:
        _start_prod_proxy(BentoMLContainer)
    finally:
        model_server_job.terminate()


def _start_dev_server(container) -> NoReturn:
    sync_container(container, BentoMLContainer)
    BentoMLContainer.model_app.get().run()
    assert False, "not reachable"


def _start_dev_proxy(container) -> NoReturn:
    sync_container(container, BentoMLContainer)
    BentoMLContainer.proxy_app.get().run()
    assert False, "not reachable"


def _start_prod_server(container) -> NoReturn:
    sync_container(container, BentoMLContainer)
    BentoMLContainer.model_server.get().run()
    assert False, "not reachable"


def _start_prod_proxy(container) -> NoReturn:
    sync_container(container, BentoMLContainer)
    BentoMLContainer.proxy_server.get().run()
    assert False, "not reachable"
