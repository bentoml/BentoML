from __future__ import annotations

import os
import sys
import json
import math
import shutil
import typing as t
import logging
import tempfile
import contextlib

import psutil
from simple_di import inject
from simple_di import Provide

from bentoml import load

from ..utils import reserve_free_port
from ..resource import CpuResource
from ..utils.uri import path_to_uri
from ..utils.circus import create_standalone_arbiter
from ..utils.analytics import track_serve
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

SCRIPT_RUNNER = "bentoml._internal.server.cli.runner"
SCRIPT_API_SERVER = "bentoml._internal.server.cli.api_server"
SCRIPT_DEV_API_SERVER = "bentoml._internal.server.cli.dev_api_server"


@inject
def ensure_prometheus_dir(
    prometheus_multiproc_dir: str = Provide[BentoMLContainer.prometheus_multiproc_dir],
    clean: bool = True,
):
    if os.path.exists(prometheus_multiproc_dir):
        if not os.path.isdir(prometheus_multiproc_dir):
            shutil.rmtree(prometheus_multiproc_dir)
        elif clean or os.listdir(prometheus_multiproc_dir):
            shutil.rmtree(prometheus_multiproc_dir)
    os.makedirs(prometheus_multiproc_dir, exist_ok=True)


@inject
def serve_development(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.api_server_config.port],
    host: str = Provide[BentoMLContainer.api_server_config.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    reload: bool = False,
) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir)  # verify service loading

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []

    circus_sockets: t.List[CircusSocket] = []
    circus_sockets.append(
        CircusSocket(
            name="_bento_api_server",
            host=host,
            port=port,
            backlog=backlog,
        )
    )

    watchers.append(
        Watcher(
            name="dev_api_server",
            cmd=sys.executable,
            args=[
                "-m",
                SCRIPT_DEV_API_SERVER,
                bento_identifier,
                "--bind",
                "fd://$(circus.sockets._bento_api_server)",
                "--working-dir",
                working_dir,
            ],
            copy_env=True,
            stop_children=True,
            use_sockets=True,
            working_dir=working_dir,
        )
    )

    plugins = []
    if reload:
        logger.debug(
            "--reload is passed. BentoML will watch file changes based on 'bentofile.yaml' and '.bentoignore' respectively."
        )

        # initialize dictionary with {} is faster than using dict()
        plugins = [
            # reloader plugin
            {
                "use": "bentoml._internal.utils.circus.watchfilesplugin.ServiceReloaderPlugin",
                "working_dir": working_dir,
                "bentoml_home": bentoml_home,
            },
        ]
    if sys.platform == "win32":
        logger.warning(
            "Due to circus limitations, output from reloader plugin will not be shown on Windows."
        )

    arbiter = create_standalone_arbiter(
        watchers,
        sockets=circus_sockets,
        plugins=plugins,
        debug=True if sys.platform != "win32" else False,
    )
    ensure_prometheus_dir()

    with track_serve(svc, production=False):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                f'Starting development BentoServer from "{bento_identifier}" '
                f"running on http://{host}:{port} (Press CTRL+C to quit)"
            ),
        )


MAX_AF_UNIX_PATH_LENGTH = 103


@inject
def serve_production(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.api_server_config.port],
    host: str = Provide[BentoMLContainer.api_server_config.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    api_workers: t.Optional[int] = None,
) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, change_global_cwd=True)

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}
    runner_bind_map: t.Dict[str, str] = {}
    uds_path = None

    if psutil.POSIX:
        # use AF_UNIX sockets for Circus
        uds_path = tempfile.mkdtemp()
        for runner in svc.runners:
            sockets_path = os.path.join(uds_path, f"{id(runner)}.sock")
            assert len(sockets_path) < MAX_AF_UNIX_PATH_LENGTH

            runner_bind_map[runner.name] = path_to_uri(sockets_path)
            circus_socket_map[runner.name] = CircusSocket(
                name=runner.name,
                path=sockets_path,
                backlog=backlog,
            )

            watchers.append(
                Watcher(
                    name=f"runner_{runner.name}",
                    cmd=sys.executable,
                    args=[
                        "-m",
                        SCRIPT_RUNNER,
                        bento_identifier,
                        "--runner-name",
                        runner.name,
                        "--bind",
                        f"fd://$(circus.sockets.{runner.name})",
                        "--working-dir",
                        working_dir,
                        "--worker-id",
                        "$(CIRCUS.WID)",
                    ],
                    copy_env=True,
                    stop_children=True,
                    working_dir=working_dir,
                    use_sockets=True,
                    numprocesses=runner.scheduled_worker_count,
                )
            )

    elif psutil.WINDOWS:
        # Windows doesn't (fully) support AF_UNIX sockets
        with contextlib.ExitStack() as port_stack:
            for runner in svc.runners:
                runner_port = port_stack.enter_context(reserve_free_port())
                runner_host = "127.0.0.1"

                runner_bind_map[runner.name] = f"tcp://{runner_host}:{runner_port}"
                circus_socket_map[runner.name] = CircusSocket(
                    name=runner.name,
                    host=runner_host,
                    port=runner_port,
                    backlog=backlog,
                )

                watchers.append(
                    Watcher(
                        name=f"runner_{runner.name}",
                        cmd=sys.executable,
                        args=[
                            "-m",
                            SCRIPT_RUNNER,
                            bento_identifier,
                            "--runner-name",
                            runner.name,
                            "--bind",
                            f"fd://$(circus.sockets.{runner.name})",
                            "--working-dir",
                            working_dir,
                            "--no-access-log",
                            "--worker-id",
                            "$(circus.wid)",
                        ],
                        copy_env=True,
                        stop_children=True,
                        use_sockets=True,
                        working_dir=working_dir,
                        numprocesses=runner.scheduled_worker_count,
                    )
                )
            port_stack.enter_context(
                reserve_free_port()
            )  # reserve one more to avoid conflicts
    else:
        raise NotImplementedError("Unsupported platform: {}".format(sys.platform))

    logger.debug("Runner map: %s", runner_bind_map)

    circus_socket_map["_bento_api_server"] = CircusSocket(
        name="_bento_api_server",
        host=host,
        port=port,
        backlog=backlog,
    )
    watchers.append(
        Watcher(
            name="api_server",
            cmd=sys.executable,
            args=[
                "-m",
                SCRIPT_API_SERVER,
                bento_identifier,
                "--bind",
                "fd://$(circus.sockets._bento_api_server)",
                "--runner-map",
                json.dumps(runner_bind_map),
                "--working-dir",
                working_dir,
                "--backlog",
                f"{backlog}",
                "--worker-id",
                "$(CIRCUS.WID)",
            ],
            copy_env=True,
            numprocesses=api_workers or math.ceil(CpuResource.from_system()),
            stop_children=True,
            use_sockets=True,
            working_dir=working_dir,
        )
    )

    arbiter = create_standalone_arbiter(
        watchers=watchers,
        sockets=list(circus_socket_map.values()),
    )

    ensure_prometheus_dir()

    with track_serve(svc, production=True):
        try:
            arbiter.start(
                cb=lambda _: logger.info(  # type: ignore
                    f'Starting production BentoServer from "{bento_identifier}" '
                    f"running on http://{host}:{port} (Press CTRL+C to quit)"
                ),
            )
        finally:
            if uds_path is not None:
                shutil.rmtree(uds_path)
