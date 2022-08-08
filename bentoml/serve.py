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
from typing import TYPE_CHECKING
from pathlib import Path

import psutil
from simple_di import inject
from simple_di import Provide

from bentoml import load
from bentoml.exceptions import UnprocessableEntity

from ._internal.log import SERVER_LOGGING_CONFIG
from ._internal.utils import reserve_free_port
from ._internal.resource import CpuResource
from ._internal.utils.uri import path_to_uri
from ._internal.utils.circus import create_standalone_arbiter
from ._internal.utils.analytics import track_serve
from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from circus.watcher import Watcher


logger = logging.getLogger(__name__)
PROMETHEUS_MESSAGE = "Prometheus metrics for {server_type} BentoServer from {bento_identifier} can be accessed at {addr}"


SCRIPT_RUNNER = "bentoml_cli.server.runner"
SCRIPT_API_SERVER = "bentoml_cli.server.http_api_server"
SCRIPT_GRPC_API_SERVER = "bentoml_cli.server.grpc_api_server"
SCRIPT_GRPC_PROMETHEUS_SERVER = "bentoml_cli.server.grpc_prometheus_server"
SCRIPT_DEV_API_SERVER = "bentoml_cli.server.http_dev_api_server"
SCRIPT_GRPC_DEV_API_SERVER = "bentoml_cli.server.grpc_dev_api_server"

MAX_AF_UNIX_PATH_LENGTH = 103

API_SERVER_NAME = "_bento_api_server"
PROMETHEUS_SERVER_NAME = "_prometheus_server"


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


def create_watcher(
    name: str,
    args: list[str],
    *,
    use_sockets: bool = True,
    **kwargs: t.Any,
) -> Watcher:
    from circus.watcher import Watcher

    return Watcher(
        name=name,
        cmd=sys.executable,
        args=args,
        copy_env=True,
        stop_children=True,
        use_sockets=use_sockets,
        **kwargs,
    )


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
    svc = load(bento_identifier, working_dir=working_dir)

    from circus.sockets import CircusSocket

    prometheus_dir = ensure_prometheus_dir()

    watchers: list[Watcher] = []

    circus_sockets: list[CircusSocket] = []
    if not grpc:
        circus_sockets.append(
            CircusSocket(name=API_SERVER_NAME, host=host, port=port, backlog=backlog)
        )

    if grpc:
        watchers.append(
            create_watcher(
                name="grpc_dev_api_server",
                args=[
                    "-m",
                    SCRIPT_GRPC_DEV_API_SERVER,
                    bento_identifier,
                    "--bind",
                    f"tcp://{host}:{port}",
                    "--working-dir",
                    working_dir,
                    "--prometheus-dir",
                    prometheus_dir,
                ],
                use_sockets=False,
                working_dir=working_dir,
            )
        )
        if BentoMLContainer.api_server_config.metrics.enabled.get():
            metrics_host = BentoMLContainer.grpc.metrics_host.get()
            metrics_port = BentoMLContainer.grpc.metrics_port.get()

            circus_sockets.append(
                CircusSocket(
                    name=PROMETHEUS_SERVER_NAME,
                    host=metrics_host,
                    port=metrics_port,
                    backlog=backlog,
                )
            )

            watchers.append(
                create_watcher(
                    name="prom_server",
                    args=[
                        "-m",
                        SCRIPT_GRPC_PROMETHEUS_SERVER,
                        "--bind",
                        f"fd://$(circus.sockets.{PROMETHEUS_SERVER_NAME})",
                        "--prometheus-dir",
                        prometheus_dir,
                        "--backlog",
                        f"{backlog}",
                    ],
                    working_dir=working_dir,
                    numprocesses=1,
                    singleton=True,
                )
            )
            logger.info(
                PROMETHEUS_MESSAGE.format(
                    bento_identifier=bento_identifier,
                    server_type="gRPC",
                    addr=f"http://{metrics_host}:{metrics_port}",
                )
            )
    else:
        watchers.append(
            create_watcher(
                name="dev_api_server",
                args=[
                    "-m",
                    SCRIPT_DEV_API_SERVER,
                    bento_identifier,
                    "--bind",
                    f"fd://$(circus.sockets.{API_SERVER_NAME})",
                    "--working-dir",
                    working_dir,
                    "--prometheus-dir",
                    prometheus_dir,
                ],
                working_dir=working_dir,
            )
        )
        logger.info(
            PROMETHEUS_MESSAGE.format(
                bento_identifier=bento_identifier,
                server_type="HTTP",
                addr=f"http://{host}:{port}/metrics",
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

        # NOTE: {} is faster than dict()
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
                f'Starting development {"HTTP" if not grpc else "gRPC"} BentoServer from "{bento_identifier}" running on http://{host}:{port} (Press CTRL+C to quit)'
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

    watchers: list[Watcher] = []
    circus_socket_map: dict[str, CircusSocket] = {}
    runner_bind_map: dict[str, str] = {}
    uds_path = None

    prometheus_dir = ensure_prometheus_dir()

    if grpc:
        if psutil.WINDOWS:
            raise UnprocessableEntity(
                "'grpc' is not supported on Windows with '--production'. The reason being SO_REUSEPORT socket option is only available on UNIX system, and gRPC implementation depends on this behaviour."
            )
        if psutil.MACOS or psutil.FREEBSD:
            logger.warning(
                f"Due to gRPC implementation on exposing SO_REUSEPORT, '--production' behaviour on {'MacOS' if psutil.MACOS else 'FreeBSD'} is not correct. We recommend to containerize BentoServer as a Linux container instead."
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
                create_watcher(
                    name=f"runner_{runner.name}",
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
                    working_dir=working_dir,
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
                    create_watcher(
                        name=f"runner_{runner.name}",
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
        watchers.append(
            create_watcher(
                name="grpc_api_server",
                args=[
                    "-m",
                    SCRIPT_GRPC_API_SERVER,
                    bento_identifier,
                    "--bind",
                    f"tcp://{host}:{port}",
                    "--runner-map",
                    json.dumps(runner_bind_map),
                    "--working-dir",
                    working_dir,
                    "--worker-id",
                    "$(CIRCUS.WID)",
                    "--prometheus-dir",
                    prometheus_dir,
                ],
                use_sockets=False,
                working_dir=working_dir,
                numprocesses=api_workers or math.ceil(CpuResource.from_system()),
            )
        )

        if BentoMLContainer.api_server_config.metrics.enabled.get():
            metrics_host = BentoMLContainer.grpc.metrics_host.get()
            metrics_port = BentoMLContainer.grpc.metrics_port.get()

            circus_socket_map[PROMETHEUS_SERVER_NAME] = CircusSocket(
                name=PROMETHEUS_SERVER_NAME,
                host=metrics_host,
                port=metrics_port,
                backlog=backlog,
            )

            watchers.append(
                create_watcher(
                    name="prom_server",
                    args=[
                        "-m",
                        SCRIPT_GRPC_PROMETHEUS_SERVER,
                        "--bind",
                        f"fd://$(circus.sockets.{PROMETHEUS_SERVER_NAME})",
                        "--prometheus-dir",
                        prometheus_dir,
                        "--backlog",
                        f"{backlog}",
                    ],
                    working_dir=working_dir,
                    numprocesses=1,
                    singleton=True,
                )
            )
            logger.info(
                PROMETHEUS_MESSAGE.format(
                    bento_identifier=bento_identifier,
                    server_type="gRPC",
                    addr=f"http://{metrics_host}:{metrics_port}",
                )
            )
    else:
        circus_socket_map[API_SERVER_NAME] = CircusSocket(
            name=API_SERVER_NAME,
            host=host,
            port=port,
            backlog=backlog,
        )

        watchers.append(
            create_watcher(
                name="api_server",
                args=[
                    "-m",
                    SCRIPT_API_SERVER,
                    bento_identifier,
                    "--bind",
                    f"fd://$(circus.sockets.{API_SERVER_NAME})",
                    "--runner-map",
                    json.dumps(runner_bind_map),
                    "--working-dir",
                    working_dir,
                    "--backlog",
                    f"{backlog}",
                    "--worker-id",
                    "$(CIRCUS.WID)",
                    "--prometheus-dir",
                    prometheus_dir,
                ],
                working_dir=working_dir,
                numprocesses=api_workers or math.ceil(CpuResource.from_system()),
            )
        )
        logger.info(
            PROMETHEUS_MESSAGE.format(
                bento_identifier=bento_identifier,
                server_type="HTTP",
                addr=f"http://{host}:{port}/metrics",
            )
        )

    arbiter = create_standalone_arbiter(
        watchers=watchers, sockets=list(circus_socket_map.values())
    )

    with track_serve(svc, production=True):
        try:
            arbiter.start(
                cb=lambda _: logger.info(  # type: ignore
                    f'Starting production {"HTTP" if not grpc else "gRPC"} BentoServer from "{bento_identifier}" running on http://{host}:{port} (Press CTRL+C to quit)'
                ),
            )
        finally:
            if uds_path is not None:
                shutil.rmtree(uds_path)
