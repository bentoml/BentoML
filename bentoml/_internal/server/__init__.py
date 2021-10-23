import logging
import sys
import tempfile
import time
from typing import Optional

from simple_di import skip

from ..bento import load_bento
from ..configuration import CONFIG_ENV_VAR, save_global_config
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)


def serve_debug(
    bento_path_or_tag: str,
    port: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    max_latency_ms: Optional[int] = None,
    run_with_ngrok: Optional[bool] = None,
    timeout: Optional[int] = None,
):
    bento = load_bento(bento_path_or_tag)

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
    bento = load_bento(bento_path_or_tag)

    import psutil

    assert (
        psutil.POSIX
    ), "BentoML API Server production mode only supports POSIX platforms"

    bento_server = BentoMLContainer.config.bento_server
    bento_server.port.set(port or skip)
    bento_server.microbatch.max_batch_size.set(max_batch_size or skip)
    bento_server.microbatch.max_latency.set(max_latency_ms or skip)

    config_file, config_path = tempfile.mkstemp(suffix="yml", text=True)
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

    for runner_name, runner in bento._runners.items():
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
    bento = load_bento(bento_path_or_tag)
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

    import json
    import os

    from circus.arbiter import Arbiter
    from circus.sockets import CircusSocket
    from circus.util import DEFAULT_ENDPOINT_DEALER, DEFAULT_ENDPOINT_SUB
    from circus.watcher import Watcher

    uds_path = tempfile.mkdtemp()

    watchers = []
    sockets_map = {}

    for runner_name, runner in bento._runners.items():
        uds_name = f"runner_{runner_name}"

        sockets_map[runner_name] = CircusSocket(
            name=uds_name, path=os.path.join(uds_path, f"{uds_name}.sock")
        )

        watchers.append(
            Watcher(
                name=f"runner_{runner_name}",
                cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_runner_server({bento_path_or_tag}, {runner_name}, instance_id=$(CIRCUS.WID), fd=$(circus.sockets.{uds_name}))"',
                env={
                    "LC_ALL": "en_US.utf-8",
                    "LANG": "en_US.utf-8",
                },
                numprocesses=runner.num_replica,
                stop_children=True,
            )
        )

    cmd_runner_arg = json.dumps(
        {k: f"$(circus.sockets.{v.name})" for k, v in sockets_map.items()}
    )
    watchers.append(
        Watcher(
            name="ngrok",
            cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_api_server({bento_path_or_tag}, instance_id=$(CIRCUS.WID), runner_fd_map={cmd_runner_arg})"',
            env={
                "LC_ALL": "en_US.utf-8",
                "LANG": "en_US.utf-8",
            },
            numprocesses=1,
            stop_children=True,
        )
    )

    """
    watchers.append(
        Watcher(
            name="dynamic_batching",
            cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_proxy({bento_path_or_tag}, instance_id=$(CIRCUS.WID))"',
            env={"LC_ALL": "en_US.utf-8", "LANG": "en_US.utf-8",},
            numprocesses=1,
            stop_children=True,
        )
    )
    """

    arbiter = Arbiter(
        watchers=watchers,
        endpoint=DEFAULT_ENDPOINT_DEALER,
        pubsub_endpoint=DEFAULT_ENDPOINT_SUB,
        sockets=[s for s in sockets_map.values()],
    )

    arbiter.start()


def _start_prod_api_server(
    bento_path_or_tag: str, instance_id: int, runner_fd_map: str
):
    bento = load_bento(bento_path_or_tag)
    bento._asgi_app()


def _start_prod_runner_server(
    bento_path_or_tag: str, name: str, instance_id: int, fd: int
):
    import uvicorn

    from bentoml._internal.server.runner_app import RunnerApp

    bento = load_bento(bento_path_or_tag)
    runner = bento._runners.get(name)

    uvicorn.run(RunnerApp(runner), fd=fd, log_level="info")
