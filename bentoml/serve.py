from __future__ import annotations

import os
import sys
import json
import math
import shutil
import socket
import typing as t
import logging
import tempfile
import contextlib
from typing import TYPE_CHECKING
from pathlib import Path
from functools import partial

import psutil
from simple_di import inject
from simple_di import Provide

from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from circus.watcher import Watcher


logger = logging.getLogger(__name__)
PROMETHEUS_MESSAGE = "Prometheus metrics for {server_type} BentoServer of '{bento_identifier}' can be accessed at '{addr}'."

SCRIPT_RUNNER = "bentoml_cli.server.runner"
SCRIPT_API_SERVER = "bentoml_cli.server.http_api_server"
SCRIPT_GRPC_API_SERVER = "bentoml_cli.server.grpc_api_server"
SCRIPT_GRPC_PROMETHEUS_SERVER = "bentoml_cli.server.grpc_prometheus_server"
SCRIPT_DEV_API_SERVER = "bentoml_cli.server.http_dev_api_server"
SCRIPT_GRPC_DEV_API_SERVER = "bentoml_cli.server.grpc_dev_api_server"

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


@contextlib.contextmanager
def enable_so_reuseport(host: str, port: int) -> t.Generator[int, None, None]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if psutil.WINDOWS:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    elif psutil.MACOS or psutil.FREEBSD:
        sock.setsockopt(socket.SOL_SOCKET, 0x10000, 1)  # SO_REUSEPORT_LB
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
            raise RuntimeError("Failed to set SO_REUSEPORT.")

    sock.bind((host, port))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


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


def log_grpcui_message(port: int) -> None:

    docker_run = partial(
        "docker run -it --rm {network_args} fullstorydev/grpcui -plaintext {platform_deps}:{port}".format,
        port=port,
    )
    message = "To use gRPC UI, run the following command: '{instruction}', followed by opening 'http://0.0.0.0:8080' in your browser of choice."

    linux_instruction = docker_run(
        platform_deps="0.0.0.0", network_args="--network=host"
    )
    mac_win_instruction = docker_run(
        platform_deps="host.docker.internal", network_args="-p 8080:8080"
    )

    if os.path.exists("/.dockerenv"):
        logger.info(
            f"If your local machine either MacOS or Windows, then use '{mac_win_instruction}', otherwise use '{linux_instruction}'."
        )
    elif psutil.WINDOWS or psutil.MACOS:
        logger.info(message.format(instruction=mac_win_instruction))
    elif psutil.LINUX:
        logger.info(message.format(instruction=linux_instruction))


def ssl_args(
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
    ssl_keyfile_password: str | None,
    ssl_version: int | None,
    ssl_cert_reqs: int | None,
    ssl_ca_certs: str | None,
    ssl_ciphers: str | None,
) -> list[str]:
    args: list[str] = []

    # Add optional SSL args if they exist
    if ssl_certfile:
        args.extend(["--ssl-certfile", str(ssl_certfile)])
    if ssl_keyfile:
        args.extend(["--ssl-keyfile", str(ssl_keyfile)])
    if ssl_keyfile_password:
        args.extend(["--ssl-keyfile-password", ssl_keyfile_password])
    if ssl_ca_certs:
        args.extend(["--ssl-ca-certs", str(ssl_ca_certs)])

    # match with default uvicorn values.
    if ssl_version:
        args.extend(["--ssl-version", str(ssl_version)])
    if ssl_cert_reqs:
        args.extend(["--ssl-cert-reqs", str(ssl_cert_reqs)])
    if ssl_ciphers:
        args.extend(["--ssl-ciphers", ssl_ciphers])
    return args


@inject
def serve_development(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.api_server_config.port],
    host: str = Provide[BentoMLContainer.api_server_config.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.http.backlog],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    ssl_certfile: str | None = Provide[BentoMLContainer.api_server_config.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.api_server_config.ssl.keyfile],
    ssl_keyfile_password: str
    | None = Provide[BentoMLContainer.api_server_config.ssl.keyfile_password],
    ssl_version: int | None = Provide[BentoMLContainer.api_server_config.ssl.version],
    ssl_cert_reqs: int
    | None = Provide[BentoMLContainer.api_server_config.ssl.cert_reqs],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.api_server_config.ssl.ca_certs],
    ssl_ciphers: str | None = Provide[BentoMLContainer.api_server_config.ssl.ciphers],
    reload: bool = False,
    grpc: bool = Provide[BentoMLContainer.grpc.enabled],
    reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
    max_concurrent_streams: int
    | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
) -> None:
    from circus.sockets import CircusSocket

    from bentoml import load

    from ._internal.log import SERVER_LOGGING_CONFIG
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir)

    prometheus_dir = ensure_prometheus_dir()

    watchers: t.List[Watcher] = []

    circus_sockets: list[CircusSocket] = []

    if grpc:
        if not reflection:
            logger.info(
                "'reflection' is disabled by default. Tools such as gRPCUI or grpcurl relies on server reflection. To use those, pass '--enable-reflection' to CLI."
            )
        else:
            log_grpcui_message(port)

        with contextlib.ExitStack() as port_stack:
            api_port = port_stack.enter_context(enable_so_reuseport(host, port))

            args = [
                "-m",
                SCRIPT_GRPC_DEV_API_SERVER,
                bento_identifier,
                "--bind",
                f"tcp://0.0.0.0:{api_port}",
                "--working-dir",
                working_dir,
            ]

            if reflection:
                args.append("--enable-reflection")
            if max_concurrent_streams:
                args.extend(
                    [
                        "--max-concurrent-streams",
                        str(max_concurrent_streams),
                    ]
                )

            watchers.append(
                create_watcher(
                    name="grpc_dev_api_server",
                    args=args,
                    use_sockets=False,
                    working_dir=working_dir,
                    # we don't want to close stdin for child process in case user use debugger.
                    # See https://circus.readthedocs.io/en/latest/for-ops/configuration/
                    close_child_stdin=False,
                )
            )

        if BentoMLContainer.api_server_config.metrics.enabled.get():
            metrics_host = BentoMLContainer.grpc.metrics.host.get()
            metrics_port = BentoMLContainer.grpc.metrics.port.get()

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
        circus_sockets.append(
            CircusSocket(name=API_SERVER_NAME, host=host, port=port, backlog=backlog)
        )

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
                    *ssl_args(
                        ssl_certfile=ssl_certfile,
                        ssl_keyfile=ssl_keyfile,
                        ssl_keyfile_password=ssl_keyfile_password,
                        ssl_version=ssl_version,
                        ssl_cert_reqs=ssl_cert_reqs,
                        ssl_ca_certs=ssl_ca_certs,
                        ssl_ciphers=ssl_ciphers,
                    ),
                ],
                working_dir=working_dir,
                # we don't want to close stdin for child process in case user use debugger.
                # See https://circus.readthedocs.io/en/latest/for-ops/configuration/
                close_child_stdin=False,
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
        if sys.platform == "win32":
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
        debug=True if sys.platform != "win32" else False,
        loggerconfig=SERVER_LOGGING_CONFIG,
        loglevel="WARNING",
    )

    with track_serve(svc, production=False):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                f'Starting development {"HTTP" if not grpc else "gRPC"} BentoServer from "{bento_identifier}" running on http://{host}:{port} (Press CTRL+C to quit)'
            ),
        )


MAX_AF_UNIX_PATH_LENGTH = 103


@inject
def serve_production(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.api_server_config.port],
    host: str = Provide[BentoMLContainer.api_server_config.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.http.backlog],
    api_workers: int | None = None,
    ssl_certfile: str | None = Provide[BentoMLContainer.api_server_config.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.api_server_config.ssl.keyfile],
    ssl_keyfile_password: str
    | None = Provide[BentoMLContainer.api_server_config.ssl.keyfile_password],
    ssl_version: int | None = Provide[BentoMLContainer.api_server_config.ssl.version],
    ssl_cert_reqs: int
    | None = Provide[BentoMLContainer.api_server_config.ssl.cert_reqs],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.api_server_config.ssl.ca_certs],
    ssl_ciphers: str | None = Provide[BentoMLContainer.api_server_config.ssl.ciphers],
    grpc: bool = Provide[BentoMLContainer.grpc.enabled],
    reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
    max_concurrent_streams: int
    | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
) -> None:
    from bentoml import load
    from bentoml.exceptions import UnprocessableEntity

    from ._internal.utils import reserve_free_port
    from ._internal.resource import CpuResource
    from ._internal.utils.uri import path_to_uri
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    from circus.sockets import CircusSocket  # type: ignore

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}
    runner_bind_map: t.Dict[str, str] = {}
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
                        "--worker-env-map",
                        json.dumps(runner.scheduled_worker_env_map),
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
                            "$(circus.wid)",
                            "--worker-env-map",
                            json.dumps(runner.scheduled_worker_env_map),
                        ],
                        working_dir=working_dir,
                        numprocesses=runner.scheduled_worker_count,
                    )
                )
            port_stack.enter_context(
                reserve_free_port()
            )  # reserve one more to avoid conflicts
    else:
        raise NotImplementedError("Unsupported platform: {}".format(sys.platform))

    logger.debug(f"Runner map: {runner_bind_map}")

    if grpc:
        if not reflection:
            logger.info(
                "'reflection' is disabled by default. Tools such as gRPCUI or grpcurl relies on server reflection. To use those, pass '--enable-reflection' to CLI."
            )
        else:
            log_grpcui_message(port)

        with contextlib.ExitStack() as port_stack:
            api_port = port_stack.enter_context(enable_so_reuseport(host, port))
            args = [
                "-m",
                SCRIPT_GRPC_API_SERVER,
                bento_identifier,
                "--bind",
                f"tcp://{host}:{api_port}",
                "--runner-map",
                json.dumps(runner_bind_map),
                "--working-dir",
                working_dir,
                "--worker-id",
                "$(CIRCUS.WID)",
            ]
            if reflection:
                args.append("--enable-reflection")

            if max_concurrent_streams:
                args.extend(
                    [
                        "--max-concurrent-streams",
                        str(max_concurrent_streams),
                    ]
                )

            watchers.append(
                create_watcher(
                    name="grpc_api_server",
                    args=args,
                    use_sockets=False,
                    working_dir=working_dir,
                    numprocesses=api_workers or math.ceil(CpuResource.from_system()),
                )
            )

        if BentoMLContainer.api_server_config.metrics.enabled.get():
            metrics_host = BentoMLContainer.grpc.metrics.host.get()
            metrics_port = BentoMLContainer.grpc.metrics.port.get()

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
                    *ssl_args(
                        ssl_certfile=ssl_certfile,
                        ssl_keyfile=ssl_keyfile,
                        ssl_keyfile_password=ssl_keyfile_password,
                        ssl_version=ssl_version,
                        ssl_cert_reqs=ssl_cert_reqs,
                        ssl_ca_certs=ssl_ca_certs,
                        ssl_ciphers=ssl_ciphers,
                    ),
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
