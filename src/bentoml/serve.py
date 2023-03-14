from __future__ import annotations

import os
import sys
import json
import shutil
import typing as t
import logging
import tempfile
import contextlib
from pathlib import Path
from functools import partial

import psutil
from simple_di import inject
from simple_di import Provide

from .exceptions import BentoMLException
from .grpc.utils import LATEST_PROTOCOL_VERSION
from ._internal.utils import experimental
from ._internal.runner.runner import Runner
from ._internal.configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    from circus.watcher import Watcher


logger = logging.getLogger(__name__)
PROMETHEUS_MESSAGE = (
    'Prometheus metrics for %s BentoServer from "%s" can be accessed at %s.'
)

SCRIPT_RUNNER = "bentoml_cli.worker.runner"
SCRIPT_API_SERVER = "bentoml_cli.worker.http_api_server"
SCRIPT_DEV_API_SERVER = "bentoml_cli.worker.http_dev_api_server"
SCRIPT_GRPC_API_SERVER = "bentoml_cli.worker.grpc_api_server"
SCRIPT_GRPC_DEV_API_SERVER = "bentoml_cli.worker.grpc_dev_api_server"
SCRIPT_GRPC_PROMETHEUS_SERVER = "bentoml_cli.worker.grpc_prometheus_server"

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
                        f"Prometheus multiproc directory {path} is not empty."
                    )
            else:
                return str(path.absolute())
        else:
            path.mkdir(parents=True)
            return str(path.absolute())
    except shutil.Error as e:
        if not use_alternative:
            raise RuntimeError(
                f"Failed to clean the Prometheus multiproc directory {directory}: {e}"
            )
    except Exception as e:  # pylint: disable=broad-except
        if not use_alternative:
            raise RuntimeError(
                f"Failed to create the Prometheus multiproc directory {directory}: {e}"
            ) from None
    try:
        alternative = tempfile.mkdtemp()
        logger.debug(
            "Failed to create the Prometheus multiproc directory %s, using temporary alternative: %s",
            directory,
            alternative,
        )
        BentoMLContainer.prometheus_multiproc_dir.set(alternative)
        return alternative
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(
            f"Failed to create temporary Prometheus multiproc directory {directory}: {e}"
        ) from None


def create_watcher(
    name: str,
    args: list[str],
    *,
    cmd: str = sys.executable,
    use_sockets: bool = True,
    **kwargs: t.Any,
) -> Watcher:
    from circus.watcher import Watcher

    return Watcher(
        name=name,
        cmd=cmd,
        args=args,
        copy_env=True,
        stop_children=True,
        use_sockets=use_sockets,
        **kwargs,
    )


def log_grpcui_instruction(port: int) -> None:
    # logs instruction on how to start gRPCUI
    docker_run = partial(
        "docker run -it --rm {network_args} fullstorydev/grpcui -plaintext {platform_deps}:{port}".format,
        port=port,
    )
    message = "To use gRPC UI, run the following command: '%s', followed by opening 'http://localhost:8080' in your browser of choice."

    linux_instruction = docker_run(
        platform_deps="localhost", network_args="--network=host"
    )
    mac_win_instruction = docker_run(
        platform_deps="host.docker.internal", network_args="-p 8080:8080"
    )

    if psutil.WINDOWS or psutil.MACOS:
        logger.info(message, mac_win_instruction)
    elif psutil.LINUX:
        logger.info(message, linux_instruction)


def construct_ssl_args(
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
    ssl_keyfile_password: str | None = None,
    ssl_version: int | None = None,
    ssl_cert_reqs: int | None = None,
    ssl_ca_certs: str | None = None,
    ssl_ciphers: str | None = None,
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


def find_triton_binary():
    binary = shutil.which("tritonserver")
    if binary is None:
        raise RuntimeError(
            "'tritonserver' is not found on PATH. Make sure to include the compiled binary in PATH to proceed.\nIf you are running this inside a container, make sure to use the official Triton container image as a 'base_image'. See https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver."
        )
    return binary


@inject
def serve_http_development(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.http.port],
    host: str = Provide[BentoMLContainer.http.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_keyfile_password: str | None = Provide[BentoMLContainer.ssl.keyfile_password],
    ssl_version: int | None = Provide[BentoMLContainer.ssl.version],
    ssl_cert_reqs: int | None = Provide[BentoMLContainer.ssl.cert_reqs],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    ssl_ciphers: str | None = Provide[BentoMLContainer.ssl.ciphers],
    reload: bool = False,
) -> None:
    prometheus_dir = ensure_prometheus_dir()

    from circus.sockets import CircusSocket

    from . import load
    from ._internal.log import SERVER_LOGGING_CONFIG
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir)

    watchers: list[Watcher] = []
    circus_sockets: list[CircusSocket] = [
        CircusSocket(name=API_SERVER_NAME, host=host, port=port, backlog=backlog)
    ]
    ssl_args = construct_ssl_args(
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_version=ssl_version,
        ssl_cert_reqs=ssl_cert_reqs,
        ssl_ca_certs=ssl_ca_certs,
        ssl_ciphers=ssl_ciphers,
    )

    watchers.append(
        create_watcher(
            name="dev_api_server",
            args=[
                "-m",
                SCRIPT_DEV_API_SERVER,
                bento_identifier,
                "--fd",
                f"$(circus.sockets.{API_SERVER_NAME})",
                "--working-dir",
                working_dir,
                "--prometheus-dir",
                prometheus_dir,
                *ssl_args,
            ],
            working_dir=working_dir,
            # we don't want to close stdin for child process in case user use debugger.
            # See https://circus.readthedocs.io/en/latest/for-ops/configuration/
            close_child_stdin=False,
        )
    )
    scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"
    if BentoMLContainer.api_server_config.metrics.enabled.get():
        log_host = "localhost" if host == "0.0.0.0" else host

        logger.info(
            PROMETHEUS_MESSAGE,
            scheme.upper(),
            bento_identifier,
            f"{scheme}://{log_host}:{port}/metrics",
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

    with track_serve(svc):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting development %s BentoServer from "%s" listening on %s://%s:%d (Press CTRL+C to quit)',
                scheme.upper(),
                bento_identifier,
                scheme,
                host,
                port,
            ),
        )


MAX_AF_UNIX_PATH_LENGTH = 103


@inject
def serve_http_production(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.http.port],
    host: str = Provide[BentoMLContainer.http.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    api_workers: int = Provide[BentoMLContainer.api_server_workers],
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_keyfile_password: str | None = Provide[BentoMLContainer.ssl.keyfile_password],
    ssl_version: int | None = Provide[BentoMLContainer.ssl.version],
    ssl_cert_reqs: int | None = Provide[BentoMLContainer.ssl.cert_reqs],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    ssl_ciphers: str | None = Provide[BentoMLContainer.ssl.ciphers],
) -> None:
    prometheus_dir = ensure_prometheus_dir()

    from circus.sockets import CircusSocket

    from . import load
    from ._internal.utils import reserve_free_port
    from ._internal.utils.uri import path_to_uri
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve
    from ._internal.configuration.containers import BentoMLContainer

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)
    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}
    runner_bind_map: t.Dict[str, str] = {}
    uds_path = None

    if psutil.POSIX:
        # use AF_UNIX sockets for Circus
        uds_path = tempfile.mkdtemp()
        for runner in svc.runners:
            if isinstance(runner, Runner):
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
                            "--fd",
                            f"$(circus.sockets.{runner.name})",
                            "--working-dir",
                            working_dir,
                            "--worker-id",
                            "$(CIRCUS.WID)",
                            "--worker-env-map",
                            json.dumps(runner.scheduled_worker_env_map),
                            "--prometheus-dir",
                            prometheus_dir,
                        ],
                        working_dir=working_dir,
                        numprocesses=runner.scheduled_worker_count,
                    )
                )
            else:
                # Make sure that the tritonserver uses the correct protocol
                runner_bind_map[runner.name] = runner.protocol_address
                cli_args = runner.cli_args + [
                    f"--http-port={runner.protocol_address.split(':')[-1]}"
                    if runner.tritonserver_type == "http"
                    else f"--grpc-port={runner.protocol_address.split(':')[-1]}"
                ]
                watchers.append(
                    create_watcher(
                        name=f"tritonserver_{runner.name}",
                        cmd=find_triton_binary(),
                        args=cli_args,
                        use_sockets=False,
                        working_dir=working_dir,
                        numprocesses=1,
                    )
                )

    elif psutil.WINDOWS:
        # Windows doesn't (fully) support AF_UNIX sockets
        with contextlib.ExitStack() as port_stack:
            for runner in svc.runners:
                if isinstance(runner, Runner):
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
                                "--fd",
                                f"$(circus.sockets.{runner.name})",
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
                else:
                    # Make sure that the tritonserver uses the correct protocol
                    runner_bind_map[runner.name] = runner.protocol_address
                    cli_args = runner.cli_args + [
                        f"--http-port={runner.protocol_address.split(':')[-1]}"
                        if runner.tritonserver_type == "http"
                        else f"--grpc-port={runner.protocol_address.split(':')[-1]}"
                    ]
                    watchers.append(
                        create_watcher(
                            name=f"tritonserver_{runner.name}",
                            cmd=find_triton_binary(),
                            args=cli_args,
                            use_sockets=False,
                            working_dir=working_dir,
                            numprocesses=1,
                        )
                    )

            # reserve one more to avoid conflicts
            port_stack.enter_context(reserve_free_port())
    else:
        raise NotImplementedError("Unsupported platform: {}".format(sys.platform))

    logger.debug("Runner map: %s", runner_bind_map)

    circus_socket_map[API_SERVER_NAME] = CircusSocket(
        name=API_SERVER_NAME,
        host=host,
        port=port,
        backlog=backlog,
    )

    ssl_args = construct_ssl_args(
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_version=ssl_version,
        ssl_cert_reqs=ssl_cert_reqs,
        ssl_ca_certs=ssl_ca_certs,
        ssl_ciphers=ssl_ciphers,
    )
    scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"
    watchers.append(
        create_watcher(
            name="api_server",
            args=[
                "-m",
                SCRIPT_API_SERVER,
                bento_identifier,
                "--fd",
                f"$(circus.sockets.{API_SERVER_NAME})",
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
                *ssl_args,
            ],
            working_dir=working_dir,
            numprocesses=api_workers,
        )
    )

    if BentoMLContainer.api_server_config.metrics.enabled.get():
        log_host = "localhost" if host == "0.0.0.0" else host

        logger.info(
            PROMETHEUS_MESSAGE,
            scheme.upper(),
            bento_identifier,
            f"{scheme}://{log_host}:{port}/metrics",
        )

    arbiter = create_standalone_arbiter(
        watchers=watchers,
        sockets=list(circus_socket_map.values()),
    )

    with track_serve(svc, production=True):
        try:
            arbiter.start(
                cb=lambda _: logger.info(  # type: ignore
                    'Starting production %s BentoServer from "%s" listening on %s://%s:%d (Press CTRL+C to quit)',
                    scheme.upper(),
                    bento_identifier,
                    scheme,
                    host,
                    port,
                ),
            )
        finally:
            if uds_path is not None:
                shutil.rmtree(uds_path)


@experimental
@inject
def serve_grpc_development(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.grpc.port],
    host: str = Provide[BentoMLContainer.grpc.host],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    max_concurrent_streams: int
    | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    reload: bool = False,
    channelz: bool = Provide[BentoMLContainer.grpc.channelz.enabled],
    reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
    protocol_version: str = LATEST_PROTOCOL_VERSION,
) -> None:
    prometheus_dir = ensure_prometheus_dir()

    from circus.sockets import CircusSocket

    from . import load
    from ._internal.log import SERVER_LOGGING_CONFIG
    from ._internal.utils import reserve_free_port
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir)

    watchers: list[Watcher] = []
    circus_sockets: list[CircusSocket] = []

    if not reflection:
        logger.info(
            "'reflection' is disabled by default. Tools such as gRPCUI or grpcurl relies on server reflection. To use those, pass '--enable-reflection' to the CLI."
        )
    else:
        log_grpcui_instruction(port)
    ssl_args = construct_ssl_args(
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_ca_certs=ssl_ca_certs,
    )

    scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"

    with contextlib.ExitStack() as port_stack:
        api_port = port_stack.enter_context(
            reserve_free_port(host, port=port, enable_so_reuseport=True)
        )

        args = [
            "-m",
            SCRIPT_GRPC_DEV_API_SERVER,
            bento_identifier,
            "--host",
            host,
            "--port",
            str(api_port),
            "--working-dir",
            working_dir,
            "--prometheus-dir",
            prometheus_dir,
            *ssl_args,
            "--protocol-version",
            protocol_version,
        ]

        if reflection:
            args.append("--enable-reflection")
        if channelz:
            args.append("--enable-channelz")
        if max_concurrent_streams:
            args.extend(
                [
                    "--max-concurrent-streams",
                    str(max_concurrent_streams),
                ]
            )

        # use circus_sockets. CircusSocket support for SO_REUSEPORT
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
                    "--fd",
                    f"$(circus.sockets.{PROMETHEUS_SERVER_NAME})",
                    "--prometheus-dir",
                    prometheus_dir,
                    "--backlog",
                    f"{backlog}",
                ],
                working_dir=working_dir,
                numprocesses=1,
                singleton=True,
                # we don't want to close stdin for child process in case user use debugger.
                # See https://circus.readthedocs.io/en/latest/for-ops/configuration/
                close_child_stdin=False,
            )
        )

        log_metrics_host = "localhost" if metrics_host == "0.0.0.0" else metrics_host

        logger.info(
            PROMETHEUS_MESSAGE,
            "gRPC",
            bento_identifier,
            f"http://{log_metrics_host}:{metrics_port}",
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
        loglevel="ERROR",
    )

    with track_serve(svc, serve_kind="grpc"):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting development %s BentoServer from "%s" listening on %s://%s:%d (Press CTRL+C to quit)',
                "gRPC",
                bento_identifier,
                scheme,
                host,
                port,
            ),
        )


@experimental
@inject
def serve_grpc_production(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.grpc.port],
    host: str = Provide[BentoMLContainer.grpc.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    api_workers: int = Provide[BentoMLContainer.api_server_workers],
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    max_concurrent_streams: int
    | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
    channelz: bool = Provide[BentoMLContainer.grpc.channelz.enabled],
    reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
    protocol_version: str = LATEST_PROTOCOL_VERSION,
) -> None:
    prometheus_dir = ensure_prometheus_dir()

    from . import load
    from ._internal.utils import reserve_free_port
    from ._internal.utils.uri import path_to_uri
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    from circus.sockets import CircusSocket  # type: ignore

    watchers: list[Watcher] = []
    circus_socket_map: dict[str, CircusSocket] = {}
    runner_bind_map: dict[str, str] = {}
    uds_path = None

    # Check whether users are running --grpc on windows
    # also raising warning if users running on MacOS or FreeBSD
    if psutil.WINDOWS:
        raise BentoMLException(
            "'grpc' is not supported on Windows with '--production'. The reason being SO_REUSEPORT socket option is only available on UNIX system, and gRPC implementation depends on this behaviour."
        )
    if psutil.MACOS or psutil.FREEBSD:
        logger.warning(
            "Due to gRPC implementation on exposing SO_REUSEPORT, '--production' behaviour on %s is not correct. We recommend to containerize BentoServer as a Linux container instead.",
            "MacOS" if psutil.MACOS else "FreeBSD",
        )

    # NOTE: We need to find and set model-repository args
    # to all TritonRunner instances (required from tritonserver if spawning multiple instances.)

    if psutil.POSIX:
        # use AF_UNIX sockets for Circus
        uds_path = tempfile.mkdtemp()
        for runner in svc.runners:
            if isinstance(runner, Runner):
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
                            "--fd",
                            f"$(circus.sockets.{runner.name})",
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
            else:
                # Make sure that the tritonserver uses the correct protocol
                runner_bind_map[runner.name] = runner.protocol_address
                cli_args = runner.cli_args + [
                    f"--http-port={runner.protocol_address.split(':')[-1]}"
                    if runner.tritonserver_type == "http"
                    else f"--grpc-port={runner.protocol_address.split(':')[-1]}"
                ]
                watchers.append(
                    create_watcher(
                        name=f"tritonserver_{runner.name}",
                        cmd=find_triton_binary(),
                        args=cli_args,
                        use_sockets=False,
                        working_dir=working_dir,
                        numprocesses=1,
                    )
                )

    elif psutil.WINDOWS:
        # Windows doesn't (fully) support AF_UNIX sockets
        with contextlib.ExitStack() as port_stack:
            for runner in svc.runners:
                if isinstance(runner, Runner):
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
                                "--fd",
                                f"$(circus.sockets.{runner.name})",
                                "--working-dir",
                                working_dir,
                                "--no-access-log",
                                "--worker-id",
                                "$(circus.wid)",
                                "--worker-env-map",
                                json.dumps(runner.scheduled_worker_env_map),
                                "--prometheus-dir",
                                prometheus_dir,
                            ],
                            working_dir=working_dir,
                            numprocesses=runner.scheduled_worker_count,
                        )
                    )
                else:
                    # Make sure that the tritonserver uses the correct protocol
                    runner_bind_map[runner.name] = runner.protocol_address
                    cli_args = runner.cli_args + [
                        f"--http-port={runner.protocol_address.split(':')[-1]}"
                        if runner.tritonserver_type == "http"
                        else f"--grpc-port={runner.protocol_address.split(':')[-1]}"
                    ]
                    watchers.append(
                        create_watcher(
                            name=f"tritonserver_{runner.name}",
                            cmd=find_triton_binary(),
                            args=cli_args,
                            use_sockets=False,
                            working_dir=working_dir,
                            numprocesses=1,
                        )
                    )
            # reserve one more to avoid conflicts
            port_stack.enter_context(reserve_free_port())
    else:
        raise NotImplementedError("Unsupported platform: {}".format(sys.platform))

    logger.debug("Runner map: %s", runner_bind_map)

    ssl_args = construct_ssl_args(
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_ca_certs=ssl_ca_certs,
    )
    scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"

    with contextlib.ExitStack() as port_stack:
        api_port = port_stack.enter_context(
            reserve_free_port(host, port=port, enable_so_reuseport=True)
        )
        args = [
            "-m",
            SCRIPT_GRPC_API_SERVER,
            bento_identifier,
            "--host",
            host,
            "--port",
            str(api_port),
            "--runner-map",
            json.dumps(runner_bind_map),
            "--working-dir",
            working_dir,
            "--prometheus-dir",
            prometheus_dir,
            "--worker-id",
            "$(CIRCUS.WID)",
            *ssl_args,
            "--protocol-version",
            protocol_version,
        ]
        if reflection:
            args.append("--enable-reflection")
        if channelz:
            args.append("--enable-channelz")
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
                numprocesses=api_workers,
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
                    "--fd",
                    f"$(circus.sockets.{PROMETHEUS_SERVER_NAME})",
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

        log_metrics_host = "127.0.0.1" if metrics_host == "0.0.0.0" else metrics_host

        logger.info(
            PROMETHEUS_MESSAGE,
            "gRPC",
            bento_identifier,
            f"http://{log_metrics_host}:{metrics_port}",
        )
    arbiter = create_standalone_arbiter(
        watchers=watchers, sockets=list(circus_socket_map.values())
    )

    with track_serve(svc, production=True, serve_kind="grpc"):
        try:
            arbiter.start(
                cb=lambda _: logger.info(  # type: ignore
                    'Starting production %s BentoServer from "%s" listening on %s://%s:%d (Press CTRL+C to quit)',
                    "gRPC",
                    bento_identifier,
                    scheme,
                    host,
                    port,
                ),
            )
        finally:
            if uds_path is not None:
                shutil.rmtree(uds_path)
