import os
import sys
import json
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
from ..utils.uri import path_to_uri
from ..utils.circus import create_standalone_arbiter
from ..utils.analytics import track_serve
from ..configuration.containers import DeploymentContainer

logger = logging.getLogger(__name__)

SCRIPT_RUNNER = "bentoml._internal.server.cli.runner"
SCRIPT_API_SERVER = "bentoml._internal.server.cli.api_server"
SCRIPT_DEV_API_SERVER = "bentoml._internal.server.cli.dev_api_server"
SCRIPT_NGROK = "bentoml._internal.server.cli.ngrok"


@inject
def ensure_prometheus_dir(
    prometheus_multiproc_dir: str = Provide[
        DeploymentContainer.prometheus_multiproc_dir
    ],
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
    port: int = Provide[DeploymentContainer.api_server_config.port],
    host: str = Provide[DeploymentContainer.api_server_config.host],
    backlog: int = Provide[DeploymentContainer.api_server_config.backlog],
    with_ngrok: bool = False,
    reload: bool = False,
    reload_delay: float = 0.25,
) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir)  # verify service loading

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    if with_ngrok:
        watchers.append(
            Watcher(
                name="ngrok",
                cmd=sys.executable,
                args=[
                    "-m",
                    SCRIPT_NGROK,
                ],
                copy_env=True,
                numprocesses=1,
                stop_children=True,
                working_dir=working_dir,
            )
        )

    circus_socket_map: t.Dict[str, CircusSocket] = {}
    circus_socket_map["_bento_api_server"] = CircusSocket(
        name="_bento_api_server",
        host=host,
        port=port,
        backlog=backlog,
    )

    watchers.append(
        Watcher(
            name="dev_api_server",
            cmd=sys.executable,
            args=[
                "-m",
                SCRIPT_DEV_API_SERVER,
                bento_identifier,
                "fd://$(circus.sockets._bento_api_server)",
                "--working-dir",
                working_dir,
            ]
            + (["--reload", "--reload-delay", f"{reload_delay}"] if reload else []),
            copy_env=True,
            numprocesses=1,
            stop_children=True,
            use_sockets=True,
            working_dir=working_dir,
        )
    )

    arbiter = create_standalone_arbiter(
        watchers,
        sockets=list(circus_socket_map.values()),
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
    port: int = Provide[DeploymentContainer.api_server_config.port],
    host: str = Provide[DeploymentContainer.api_server_config.host],
    backlog: int = Provide[DeploymentContainer.api_server_config.backlog],
    app_workers: t.Optional[int] = None,
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
        for runner_name, runner in svc.runners.items():
            sockets_path = os.path.join(uds_path, f"{id(runner)}.sock")
            assert len(sockets_path) < MAX_AF_UNIX_PATH_LENGTH

            runner_bind_map[runner_name] = path_to_uri(sockets_path)
            circus_socket_map[runner_name] = CircusSocket(
                name=runner_name,
                path=sockets_path,
                backlog=backlog,
            )

            watchers.append(
                Watcher(
                    name=f"runner_{runner_name}",
                    cmd=sys.executable,
                    args=[
                        "-m",
                        SCRIPT_RUNNER,
                        bento_identifier,
                        runner_name,
                        f"fd://$(circus.sockets.{runner_name})",
                        "--working-dir",
                        working_dir,
                    ],
                    copy_env=True,
                    stop_children=True,
                    working_dir=working_dir,
                    use_sockets=True,
                    numprocesses=runner.num_replica,
                )
            )

    elif psutil.WINDOWS:
        # Windows doesn't (fully) support AF_UNIX sockets
        with contextlib.ExitStack() as port_stack:
            for runner_name, runner in svc.runners.items():
                runner_port = port_stack.enter_context(reserve_free_port())
                runner_host = "127.0.0.1"

                runner_bind_map[runner_name] = f"tcp://{runner_host}:{runner_port}"
                circus_socket_map[runner_name] = CircusSocket(
                    name=runner_name,
                    host=runner_host,
                    port=runner_port,
                    backlog=backlog,
                )

                watchers.append(
                    Watcher(
                        name=f"runner_{runner_name}",
                        cmd=sys.executable,
                        args=[
                            "-m",
                            SCRIPT_RUNNER,
                            bento_identifier,
                            runner_name,
                            f"fd://$(circus.sockets.{runner_name})",
                            "--working-dir",
                            working_dir,
                        ],
                        copy_env=True,
                        stop_children=True,
                        use_sockets=True,
                        working_dir=working_dir,
                        numprocesses=runner.num_replica,
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
                "fd://$(circus.sockets._bento_api_server)",
                "--runner-map",
                json.dumps(runner_bind_map),
                "--working-dir",
                working_dir,
                "--backlog",
                f"{backlog}",
            ],
            copy_env=True,
            numprocesses=app_workers or 1,
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
                    f'Starting production BentoServer from "bento_identifier" '
                    f"running on http://{host}:{port} (Press CTRL+C to quit)"
                ),
            )
        finally:
            if uds_path is not None:
                shutil.rmtree(uds_path)
