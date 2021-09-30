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
import sys
import tempfile
import time
from typing import Optional

from simple_di import skip

from bentoml._internal.configuration import CONFIG_ENV_VAR, save_global_config
from bentoml._internal.configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)


def serve_debug(
    bento_path_or_tag: str,
    port: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    max_latency_ms: Optional[int] = None,
    run_with_ngrok: Optional[bool] = None,
    timeout: Optional[int] = None,
):
    bento = load_from_dir(bento_path_or_tag)

    bento_server = BentoMLContainer.config.bento_server
    bento_server.port.set(port or skip)
    bento_server.timeout.set(timeout or skip)
    bento_server.microbatch.timeout.set(timeout or skip)
    bento_server.microbatch.max_batch_size.set(max_batch_size or skip)
    bento_server.microbatch.max_latency.set(max_latency_ms or skip)

    BentoMLContainer.forward_port.get()  # generate port before fork

    from circus.arbiter import Arbiter
    from circus.util import DEFAULT_ENDPOINT_DEALER, DEFAULT_ENDPOINT_SUB
    from circus.watcher import Watcher

    watchers = []

    if run_with_ngrok:
        watchers.append(
            Watcher(
                name="ngrok",
                cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_ngrok_server()"',
                env={
                    "LC_ALL": "en_US.utf-8",
                    "LANG": "en_US.utf-8",
                },
                numprocesses=1,
                stop_children=True,
            )
        )

    watchers.append(
        Watcher(
            name="ngrok",
            cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_dev_server()"',
            env={
                "LC_ALL": "en_US.utf-8",
                "LANG": "en_US.utf-8",
            },
            numprocesses=1,
            stop_children=True,
        )
    )
    watchers.append(
        Watcher(
            name="ngrok",
            cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_dev_proxy()"',
            env={
                "LC_ALL": "en_US.utf-8",
                "LANG": "en_US.utf-8",
            },
            numprocesses=1,
            stop_children=True,
        )
    )

    arbiter = Arbiter(
        watchers=watchers,
        endpoint=DEFAULT_ENDPOINT_DEALER,
        pubsub_endpoint=DEFAULT_ENDPOINT_SUB,
    )

    arbiter.start()


def serve(
    bento_path_or_tag: str,
    port: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    max_latency_ms: Optional[int] = None,
):
    bento = load_from_dir(bento_path_or_tag)

    import psutil

    assert (
        psutil.POSIX
    ), "BentoML API Server production mode only supports POSIX platforms"

    bento_server = BentoMLContainer.config.bento_server
    bento_server.port.set(port or skip)
    bento_server.microbatch.max_batch_size.set(max_batch_size or skip)
    bento_server.microbatch.max_latency.set(max_latency_ms or skip)

    config_file, config_path = tempfile.mkstemp(suffix='yml', text=True)
    save_global_config(config_file)  # save the container state to yml file

    from circus.arbiter import Arbiter
    from circus.util import DEFAULT_ENDPOINT_DEALER, DEFAULT_ENDPOINT_SUB
    from circus.watcher import Watcher

    watchers = []

    watchers.append(
        Watcher(
            name="http server",
            cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_server({bento_path_or_tag})"',
            env={
                "LC_ALL": "en_US.utf-8",
                "LANG": "en_US.utf-8",
                CONFIG_ENV_VAR: config_path,
            },
            numprocesses=1,
            stop_children=True,
        )
    )

    watchers.append(
        Watcher(
            name="marshal",
            cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_proxy({bento_path_or_tag})"',
            env={
                "LC_ALL": "en_US.utf-8",
                "LANG": "en_US.utf-8",
                CONFIG_ENV_VAR: config_path,
            },
            numprocesses=1,
            stop_children=True,
        )
    )

    for runner_name, runner in bento.runners.items():
        watchers.append(
            Watcher(
                name="ngrok",
                cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_proxy({bento_path_or_tag})"',
                env={
                    "LC_ALL": "en_US.utf-8",
                    "LANG": "en_US.utf-8",
                    CONFIG_ENV_VAR: config_path,
                },
                numprocesses=1,
                stop_children=True,
            )
        )

    arbiter = Arbiter(
        watchers=watchers,
        endpoint=DEFAULT_ENDPOINT_DEALER,
        pubsub_endpoint=DEFAULT_ENDPOINT_SUB,
    )

    arbiter.start()


def _start_ngrok_server():
    from bentoml.utils.flask_ngrok import start_ngrok

    time.sleep(1)
    start_ngrok(BentoMLContainer.config.bento_server.port.get())


def _start_dev_server():
    BentoMLContainer.model_app.get().run()


def _start_dev_proxy():
    BentoMLContainer.proxy_app.get().run()


def _start_prod_server():
    BentoMLContainer.model_server.get().run()


def _start_prod_proxy():
    BentoMLContainer.proxy_server.get().run()


def start_prod_server1(
    bento_path_or_tag: str,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    timeout: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    max_latency_ms: Optional[int] = None,
):
    bento = load_from_dir(bento_path_or_tag)
    import psutil

    assert (
        psutil.POSIX
    ), "BentoML API Server production mode only supports POSIX platforms"

    bento_server = BentoMLContainer.config.bento_server
    bento_server.port.set(port or skip)
    bento_server.timeout.set(timeout or skip)
    bento_server.microbatch.timeout.set(timeout or skip)
    bento_server.workers.set(workers or skip)
    bento_server.microbatch.max_batch_size.set(max_batch_size or skip)
    bento_server.microbatch.max_latency.set(max_latency_ms or skip)

    BentoMLContainer.forward_port.get()  # generate port before fork

    from circus.arbiter import Arbiter
    from circus.util import DEFAULT_ENDPOINT_DEALER, DEFAULT_ENDPOINT_SUB
    from circus.watcher import Watcher

    watchers = []

    watchers.append(
        Watcher(
            name="ngrok",
            cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_api_server()"',
            env={
                "LC_ALL": "en_US.utf-8",
                "LANG": "en_US.utf-8",
            },
            numprocesses=1,
            stop_children=True,
        )
    )

    watchers.append(
        Watcher(
            name="ngrok",
            cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_proxy()"',
            env={
                "LC_ALL": "en_US.utf-8",
                "LANG": "en_US.utf-8",
            },
            numprocesses=1,
            stop_children=True,
        )
    )

    for runner in bento.runners.values():
        watchers.append(
            Watcher(
                name="ngrok",
                cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_runner_server({runner.name})"',
                env={
                    "LC_ALL": "en_US.utf-8",
                    "LANG": "en_US.utf-8",
                },
                numprocesses=runner.replicas,
                stop_children=True,
            )
        )

    arbiter = Arbiter(
        watchers=watchers,
        endpoint=DEFAULT_ENDPOINT_DEALER,
        pubsub_endpoint=DEFAULT_ENDPOINT_SUB,
    )

    arbiter.start()


def _start_prod_api_server():
    BentoMLContainer.api_server.get().run()


def _start_prod_runner_server(name, fd):
    import uvicorn

    runner = BentoMLContainer.service.get_runner_by_name(name)

    uvicorn.run(runner._asgi_app, fd=fd, log_level="info")
