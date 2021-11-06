import logging
import os
import sys
import tempfile
import time
import typing as t

from bentoml import load
from bentoml._internal.configuration import get_debug_mode
from bentoml._internal.configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)


def serve_development(
    bento_path_or_tag: str,
    working_dir: t.Optional[str] = None,
    port: t.Optional[int] = None,
    with_ngrok: bool = False,
    reload: bool = False,
    reload_delay: float = 0.25,
) -> None:
    if working_dir is not None:
        working_dir = os.path.abspath(working_dir)

    svc = load(bento_path_or_tag, working_dir=working_dir)

    from circus.arbiter import Arbiter
    from circus.util import DEFAULT_ENDPOINT_DEALER, DEFAULT_ENDPOINT_SUB
    from circus.watcher import Watcher

    env = dict(os.environ)

    watchers = []
    for _, runner in svc._runners.items():
        runner._setup()

    if with_ngrok:
        watchers.append(
            Watcher(
                name="ngrok",
                cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_ngrok_server()"',
                env=env,
                numprocesses=1,
                stop_children=True,
            )
        )

    watchers.append(
        Watcher(
            name="ngrok",
            cmd=f'{sys.executable} -c \'import bentoml._internal.server; bentoml._internal.server._start_dev_api_server("{bento_path_or_tag}", {port}, "{working_dir}", reload={reload}, reload_delay={reload_delay}, instance_id=$(CIRCUS.WID))\'',
            env=env,
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


def _start_ngrok_server() -> None:
    from bentoml._internal.utils.flask_ngrok import start_ngrok

    time.sleep(1)
    start_ngrok(BentoMLContainer.config.bento_server.port.get())


def serve_production(
    bento_path_or_tag: str,
    working_dir: t.Optional[str] = None,
    app_workers: t.Optional[int] = None,
    runner_workers: t.Optional[int] = None,
    port: t.Optional[int] = None,
) -> None:
    svc = load(bento_path_or_tag, working_dir=working_dir)

    env = dict(os.environ)

    import json

    from circus.arbiter import Arbiter
    from circus.sockets import CircusSocket
    from circus.util import DEFAULT_ENDPOINT_DEALER, DEFAULT_ENDPOINT_SUB
    from circus.watcher import Watcher

    uds_path = tempfile.mkdtemp()

    watchers = []
    sockets_map = {}

    for runner_name, runner in svc._runners.items():
        uds_name = f"runner_{runner_name}"

        sockets_map[runner_name] = CircusSocket(
            name=uds_name, path=os.path.join(uds_path, f"{uds_name}.sock")
        )

        watchers.append(
            Watcher(
                name=f"runner_{runner_name}",
                cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_runner_server({bento_path_or_tag}, {runner_name}, instance_id=$(CIRCUS.WID), fd=$(circus.sockets.{uds_name}))"',
                env=env,
                numprocesses=runner.num_replica,
                stop_children=True,
            )
        )

    cmd_runner_arg = json.dumps(
        {k: f"$(circus.sockets.{v.name})" for k, v in sockets_map.items()}
    )
    watchers.append(
        Watcher(
            name="api_server",
            cmd=f'{sys.executable} -c "import bentoml; bentoml.server._start_prod_api_server({bento_path_or_tag}, instance_id=$(CIRCUS.WID), runner_fd_map={cmd_runner_arg})"',
            env=env,
            numprocesses=1,
            stop_children=True,
        )
    )

    arbiter = Arbiter(
        watchers=watchers,
        endpoint=DEFAULT_ENDPOINT_DEALER,
        pubsub_endpoint=DEFAULT_ENDPOINT_SUB,
        sockets=[s for s in sockets_map.values()],
    )

    arbiter.start()


serve_production = serve_development  # TODO(jiang): remove me # noqa: F811


def _start_dev_api_server(
    bento_path_or_tag: str,
    port: int,
    working_dir: str,
    reload: bool = False,
    reload_delay: t.Optional[float] = None,
    instance_id: t.Optional[int] = None,
):
    import uvicorn
    from uvicorn.config import Config
    from uvicorn.server import Server
    from uvicorn.supervisors import ChangeReload

    log_level = "debug" if get_debug_mode() else "info"
    svc = load(bento_path_or_tag, working_dir=working_dir)
    uvicorn_options = {
        "port": port,
        "log_level": log_level,
        "reload": reload,
        "reload_delay": reload_delay,
    }

    if reload:
        asgi_app_import_str = f"{svc._import_str}.asgi_app"
        config = Config(asgi_app_import_str, **uvicorn_options)
        server = Server(config=config)
        sock = config.bind_socket()
        # TODO: use svc.build_args.include/exclude as default files to watch
        # TODO: watch changes in model store when "latest" model tag is used
        ChangeReload(config, target=server.run, sockets=[sock]).run()
    else:
        uvicorn.run(svc.asgi_app, **uvicorn_options)


"""
def _start_prod_api_server(bento_path_or_tag: str, instance_id: int, runners_map: str):
    import uvicorn

    svc = load(bento_path_or_tag)

    uvicorn.run(svc.asgi_app, log_level="info")


def _start_prod_runner_server(
    bento_path_or_tag: str, name: str, instance_id: int, fd: int
):
    import uvicorn

    from bentoml._internal.server.runner_app import RunnerApp

    svc = load(bento_path_or_tag)
    runner = svc._runners.get(name)

    uvicorn.run(RunnerApp(runner), fd=fd, log_level="info")
"""
