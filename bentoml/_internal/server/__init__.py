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
from pathlib import Path

import psutil
from simple_di import inject
from simple_di import Provide

from bentoml import load
from bentoml.exceptions import UnprocessableEntity

from ..log import SERVER_LOGGING_CONFIG
from ..utils import reserve_free_port
from ..resource import CpuResource
from ..utils.uri import path_to_uri
from ..utils.circus import create_standalone_arbiter
from ..utils.analytics import track_serve
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)


SCRIPT_RUNNER = "bentoml._internal.server.cli.runner"
SCRIPT_API_SERVER = "bentoml._internal.server.cli.http_api_server"
SCRIPT_GRPC_API_SERVER = "bentoml._internal.server.cli.grpc_api_server"
SCRIPT_DEV_API_SERVER = "bentoml._internal.server.cli.http_dev_api_server"
SCRIPT_GRPC_DEV_API_SERVER = "bentoml._internal.server.cli.grpc_dev_api_server"

MAX_AF_UNIX_PATH_LENGTH = 103

API_SERVER_NAME = "_bento_api_server"


@inject
def ensure_prometheus_dir(
    directory: str = Provide[BentoMLContainer.prometheus_multiproc_dir],
    clean: bool = True,
    use_alternative: bool = True,
) -> str:
    try:
        path = Path(directory)
        if path.exists():
            if not path.is_dir() or any(path.iterdir()):
                if clean:
                    shutil.rmtree(str(path))
                    path.mkdir()
                    return str(path.absolute())
                else:
                    raise RuntimeError(
                        "Prometheus multiproc directory {} is not empty".format(path)
                    )
            else:
                return str(path.absolute())
        else:
            path.mkdir(parents=True)
            return str(path.absolute())
    except shutil.Error as e:
        if not use_alternative:
            raise RuntimeError(
                f"Failed to clean the prometheus multiproc directory {directory}: {e}"
            )
    except OSError as e:
        if not use_alternative:
            raise RuntimeError(
                f"Failed to create the prometheus multiproc directory {directory}: {e}"
            )
    assert use_alternative
    alternative = tempfile.mkdtemp()
    logger.warning(
        f"Failed to ensure the prometheus multiproc directory {directory}, "
        f"using alternative: {alternative}",
    )
    BentoMLContainer.prometheus_multiproc_dir.set(alternative)
    return alternative


@inject
def serve_development(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.api_server_config.port],
    host: str = Provide[BentoMLContainer.api_server_config.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    reload: bool = False,
    grpc: bool = False,
) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir)  # verify service loading

    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

    prometheus_dir = ensure_prometheus_dir()

    watchers: list[Watcher] = []

    if grpc:
        watcher_name = "grpc_dev_api_server"
        script_to_use = SCRIPT_GRPC_DEV_API_SERVER
        bind_address = f"tcp://{host}:{port}"
    else:
        watcher_name = "dev_api_server"
        script_to_use = SCRIPT_DEV_API_SERVER
        bind_address = f"fd://$(circus.sockets.{API_SERVER_NAME})"

    circus_sockets: list[CircusSocket] = []
    circus_sockets.append(
        CircusSocket(name=API_SERVER_NAME, host=host, port=port, backlog=backlog)
    )

    watchers.append(
        Watcher(
            name=watcher_name,
            cmd=sys.executable,
            args=[
                "-m",
                script_to_use,
                bento_identifier,
                "--bind",
                bind_address,
                "--working-dir",
                working_dir,
                "--prometheus-dir",
                prometheus_dir,
            ],
            copy_env=True,
            stop_children=True,
            use_sockets=True,
            working_dir=working_dir,
        )
    )

    plugins = []
    if reload:
        if psutil.WINDOWS:
            logger.warning(
                "Due to circus limitations, output from the reloader plugin will not be shown on Windows."
            )
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
    arbiter = create_standalone_arbiter(
        watchers,
        sockets=circus_sockets,
        plugins=plugins,
        debug=not psutil.WINDOWS,
        loggerconfig=SERVER_LOGGING_CONFIG,
        loglevel="WARNING",
    )

    with track_serve(svc, production=False):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                f'Starting development BentoServer from "{bento_identifier}" '
                f"running on http://{host}:{port} (Press CTRL+C to quit)"
            ),
        )


@inject
def serve_production(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.api_server_config.port],
    host: str = Provide[BentoMLContainer.api_server_config.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    max_concurrent_streams: int = Provide[BentoMLContainer.grpc.max_concurrent_streams],
    api_workers: t.Optional[int] = None,
    grpc: bool = False,
) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

    watchers: list[Watcher] = []
    circus_socket_map: dict[str, CircusSocket] = {}
    runner_bind_map: dict[str, str] = {}
    uds_path = None

    prometheus_dir = ensure_prometheus_dir()

    if grpc and psutil.WINDOWS:
        raise UnprocessableEntity(
            "'grpc' is not supported on Windows with '--production'. The reason being SO_REUSEPORT socket option is only available on UNIX system, and gRPC implementation depends on this behaviour."
        )

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
                            "$(CIRCUS.WID)",
                        ],
                        copy_env=True,
                        stop_children=True,
                        use_sockets=True,
                        working_dir=working_dir,
                        numprocesses=runner.scheduled_worker_count,
                    )
                )
            # reserve one more to avoid conflicts
            port_stack.enter_context(reserve_free_port())
    else:
        raise NotImplementedError("Unsupported platform: {}".format(sys.platform))

    logger.debug("Runner map: %s", runner_bind_map)

    if grpc:
        watcher_name = "grpc_api_server"
        script_to_use = SCRIPT_GRPC_API_SERVER
        socket_path = f"tcp://{host}:{port}"
        # num_connect_args = ["--max-concurrent-streams", f"{max_concurrent_streams}"]
        num_connect_args = []
    else:
        watcher_name = "api_server"
        script_to_use = SCRIPT_API_SERVER
        socket_path = f"fd://$(circus.sockets.{API_SERVER_NAME})"
        num_connect_args = ["--backlog", f"{backlog}"]

    circus_socket_map[API_SERVER_NAME] = CircusSocket(
        name=API_SERVER_NAME,
        host=host,
        port=port,
        backlog=backlog,
    )

    watchers.append(
        Watcher(
            name=watcher_name,
            cmd=sys.executable,
            args=[
                "-m",
                script_to_use,
                bento_identifier,
                "--bind",
                socket_path,
                "--runner-map",
                json.dumps(runner_bind_map),
                "--working-dir",
                working_dir,
                *num_connect_args,
                "--worker-id",
                "$(CIRCUS.WID)",
                "--prometheus-dir",
                prometheus_dir,
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
