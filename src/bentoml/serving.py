from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import shlex
import shutil
import sys
import tempfile
import typing as t
from functools import partial
from pathlib import Path

import psutil
from simple_di import Provide
from simple_di import inject

from bentoml._internal.log import SERVER_LOGGING_CONFIG
from bentoml._internal.utils.circus import Server

from ._internal.configuration.containers import BentoMLContainer
from ._internal.runner.runner import Runner
from .exceptions import BentoMLConfigException
from .exceptions import BentoMLException
from .grpc.utils import LATEST_PROTOCOL_VERSION

if t.TYPE_CHECKING:
    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

    from ._internal.service import Service


logger = logging.getLogger(__name__)
PROMETHEUS_MESSAGE = (
    'Prometheus metrics for %s BentoServer from "%s" can be accessed at %s.'
)

SCRIPT_RUNNER = "bentoml_cli.worker.runner"
SCRIPT_API_SERVER = "bentoml_cli.worker.http_api_server"
SCRIPT_GRPC_API_SERVER = "bentoml_cli.worker.grpc_api_server"
SCRIPT_GRPC_PROMETHEUS_SERVER = "bentoml_cli.worker.grpc_prometheus_server"

API_SERVER_NAME = "_bento_api_server"
PROMETHEUS_SERVER_NAME = "_prometheus_server"
IS_WSL = "microsoft-standard" in platform.release()


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
        cmd=shlex.quote(cmd) if psutil.POSIX else cmd,
        args=args,
        copy_env=True,
        stop_children=True,
        use_sockets=use_sockets,
        graceful_timeout=86400,
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


def construct_timeouts_args(
    timeout_keep_alive: int | None,
    timeout_graceful_shutdown: int | None,
) -> list[str]:
    args: list[str] = []

    if timeout_keep_alive:
        args.extend(["--timeout-keep-alive", str(timeout_keep_alive)])
    if timeout_graceful_shutdown:
        args.extend(["--timeout-graceful-shutdown", str(timeout_graceful_shutdown)])

    return args


def find_triton_binary():
    binary = shutil.which("tritonserver")
    if binary is None:
        raise RuntimeError(
            "'tritonserver' is not found on PATH. Make sure to include the compiled binary in PATH to proceed.\nIf you are running this inside a container, make sure to use the official Triton container image as a 'base_image'. See https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver."
        )
    return binary


def make_reload_plugin(working_dir: str, bentoml_home: str) -> dict[str, str]:
    if sys.platform == "win32":
        logger.warning(
            "Due to circus limitations, output from the reloader plugin will not be shown on Windows."
        )
    logger.debug(
        "reload is enabled. BentoML will watch file changes based on 'bentofile.yaml' and '.bentoignore' respectively."
    )

    return {
        "use": "bentoml._internal.utils.circus.watchfilesplugin.ServiceReloaderPlugin",
        "working_dir": working_dir,
    }


def on_service_deployment(service: Service) -> None:
    for on_deployment in service.deployment_hooks:
        on_deployment()


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
    timeout_keep_alive: int | None = None,
    timeout_graceful_shutdown: int | None = None,
    threaded: bool = False,
) -> Server:
    logger.warning(
        "serve_http_development is deprecated. Please use serve_http_production with api_workers=1 and development_mode=True"
    )

    return serve_http_production(
        bento_identifier,
        working_dir,
        port=port,
        host=host,
        backlog=backlog,
        bentoml_home=bentoml_home,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_version=ssl_version,
        ssl_cert_reqs=ssl_cert_reqs,
        ssl_ca_certs=ssl_ca_certs,
        ssl_ciphers=ssl_ciphers,
        reload=reload,
        api_workers=1,
        development_mode=True,
        timeout_keep_alive=timeout_keep_alive,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
        threaded=threaded,
    )


MAX_AF_UNIX_PATH_LENGTH = 103


def _get_runner_socket_posix(
    runner: Runner, uds_path: str | None, port_stack: contextlib.ExitStack, backlog: int
) -> tuple[str, CircusSocket]:
    from circus.sockets import CircusSocket

    from ._internal.utils.uri import path_to_uri

    assert uds_path is not None
    socket_path = os.path.join(uds_path, f"{id(runner)}.sock")
    assert len(socket_path) < MAX_AF_UNIX_PATH_LENGTH
    return path_to_uri(socket_path), CircusSocket(
        name=runner.name,
        path=socket_path,
        backlog=backlog,
    )


def _get_runner_socket_windows(
    runner: Runner, uds_path: str | None, port_stack: contextlib.ExitStack, backlog: int
) -> tuple[str, CircusSocket]:
    from circus.sockets import CircusSocket

    from ._internal.utils import reserve_free_port

    runner_port = port_stack.enter_context(reserve_free_port())
    runner_host = "127.0.0.1"

    return f"tcp://{runner_host}:{runner_port}", CircusSocket(
        name=runner.name,
        host=runner_host,
        port=runner_port,
        backlog=backlog,
    )


@inject(squeeze_none=True)
def serve_http_production(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.http.port],
    host: str = Provide[BentoMLContainer.http.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    api_workers: int = Provide[BentoMLContainer.api_server_workers],
    timeout: int | None = None,
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_keyfile_password: str | None = Provide[BentoMLContainer.ssl.keyfile_password],
    ssl_version: int | None = Provide[BentoMLContainer.ssl.version],
    ssl_cert_reqs: int | None = Provide[BentoMLContainer.ssl.cert_reqs],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    ssl_ciphers: str | None = Provide[BentoMLContainer.ssl.ciphers],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    development_mode: bool = False,
    reload: bool = False,
    timeout_keep_alive: int | None = None,
    timeout_graceful_shutdown: int | None = None,
    threaded: bool = False,
) -> Server:
    env = {"PROMETHEUS_MULTIPROC_DIR": ensure_prometheus_dir()}

    import ipaddress
    from socket import AF_INET
    from socket import AF_INET6

    from circus.sockets import CircusSocket

    from . import load
    from ._internal.configuration.containers import BentoMLContainer
    from ._internal.utils import reserve_free_port
    from ._internal.utils.analytics import track_serve
    from ._internal.utils.circus import create_standalone_arbiter

    svc = load(bento_identifier, working_dir=working_dir)
    working_dir = os.path.realpath(os.path.expanduser(working_dir))

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}
    runner_bind_map: t.Dict[str, str] = {}
    uds_path = None
    timeout_args = ["--timeout", str(timeout)] if timeout else []

    if psutil.POSIX and not IS_WSL:
        # use AF_UNIX sockets for Circus
        uds_path = tempfile.mkdtemp()
        get_socket_func = _get_runner_socket_posix
    elif psutil.WINDOWS or IS_WSL:
        get_socket_func = _get_runner_socket_windows
    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")
    with contextlib.ExitStack() as port_stack:
        for runner in svc.runners:
            if isinstance(runner, Runner):
                runner_args = [
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
                    *timeout_args,
                ]
                if runner.embedded or development_mode:
                    continue

                socket_uri, circus_socket = get_socket_func(
                    runner, uds_path, port_stack, backlog
                )

                runner_bind_map[runner.name] = socket_uri
                circus_socket_map[runner.name] = circus_socket

                watchers.append(
                    create_watcher(
                        name=f"runner_{runner.name}",
                        args=runner_args,
                        working_dir=working_dir,
                        numprocesses=runner.scheduled_worker_count,
                        env=env,
                    )
                )
            else:
                # Make sure that the tritonserver uses the correct protocol
                runner_bind_map[runner.name] = runner.protocol_address
                cli_args = runner.cli_args + [
                    (
                        f"--http-port={runner.protocol_address.split(':')[-1]}"
                        if runner.tritonserver_type == "http"
                        else f"--grpc-port={runner.protocol_address.split(':')[-1]}"
                    )
                ]
                watchers.append(
                    create_watcher(
                        name=f"tritonserver_{runner.name}",
                        cmd=find_triton_binary(),
                        args=cli_args,
                        use_sockets=False,
                        working_dir=working_dir,
                        numprocesses=1,
                        env=env,
                    )
                )
        # reserve one more to avoid conflicts
        port_stack.enter_context(reserve_free_port())

    logger.debug("Runner map: %s", runner_bind_map)

    try:
        ipaddr = ipaddress.ip_address(host)
        if ipaddr.version == 4:
            family = AF_INET
        elif ipaddr.version == 6:
            family = AF_INET6
        else:
            raise BentoMLConfigException(
                f"Unsupported host IP address version: {ipaddr.version}"
            )
    except ValueError as e:
        raise BentoMLConfigException(f"Invalid host IP address: {host}") from e

    circus_socket_map[API_SERVER_NAME] = CircusSocket(
        name=API_SERVER_NAME,
        host=host,
        port=port,
        family=family,
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

    timeouts_args = construct_timeouts_args(
        timeout_keep_alive=timeout_keep_alive,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
    )

    api_server_args = [
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
        *ssl_args,
        *timeouts_args,
        *timeout_args,
    ]

    if development_mode:
        api_server_args.append("--development-mode")

    close_child_stdin = False if development_mode else True

    on_service_deployment(svc)

    scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"
    watchers.append(
        create_watcher(
            name="api_server",
            args=api_server_args,
            working_dir=working_dir,
            numprocesses=api_workers,
            close_child_stdin=close_child_stdin,
            env=env,
        )
    )

    log_host = "localhost" if host in ["0.0.0.0", "::"] else host
    if BentoMLContainer.api_server_config.metrics.enabled.get():
        logger.info(
            PROMETHEUS_MESSAGE,
            scheme.upper(),
            bento_identifier,
            f"{scheme}://{log_host}:{port}/metrics",
        )

    arbiter_kwargs: dict[str, t.Any] = {
        "watchers": watchers,
        "sockets": list(circus_socket_map.values()),
        "threaded": threaded,
    }

    plugins = []

    if reload:
        reload_plugin = make_reload_plugin(working_dir, bentoml_home)
        plugins.append(reload_plugin)

    arbiter_kwargs["plugins"] = plugins

    if development_mode:
        arbiter_kwargs["debug"] = True if sys.platform != "win32" else False
        arbiter_kwargs["loggerconfig"] = SERVER_LOGGING_CONFIG
        arbiter_kwargs["loglevel"] = "WARNING"

    arbiter = create_standalone_arbiter(**arbiter_kwargs)

    production = False if development_mode else True
    arbiter.exit_stack.enter_context(
        track_serve(svc, production=production, serve_kind="http")
    )

    @arbiter.exit_stack.callback
    def cleanup():
        if uds_path is not None:
            shutil.rmtree(uds_path)

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
        return Server(url=f"{scheme}://{log_host}:{port}", arbiter=arbiter)
    except Exception:
        cleanup()
        raise


@inject(squeeze_none=True)
def serve_grpc_production(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.grpc.port],
    host: str = Provide[BentoMLContainer.grpc.host],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    api_workers: int = Provide[BentoMLContainer.api_server_workers],
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    max_concurrent_streams: int | None = Provide[
        BentoMLContainer.grpc.max_concurrent_streams
    ],
    channelz: bool = Provide[BentoMLContainer.grpc.channelz.enabled],
    reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
    protocol_version: str = LATEST_PROTOCOL_VERSION,
    reload: bool = False,
    development_mode: bool = False,
    threaded: bool = False,
) -> Server:
    env = {"PROMETHEUS_MULTIPROC_DIR": ensure_prometheus_dir()}

    from . import load
    from ._internal.service import Service
    from ._internal.utils import reserve_free_port
    from ._internal.utils.analytics import track_serve
    from ._internal.utils.circus import create_standalone_arbiter

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir)

    if not isinstance(svc, Service):
        raise BentoMLException(f"{type(svc)} type doesn't support gRPC serving")

    from circus.sockets import CircusSocket  # type: ignore

    watchers: list[Watcher] = []
    circus_socket_map: dict[str, CircusSocket] = {}
    runner_bind_map: dict[str, str] = {}
    uds_path = None

    # Check whether users are running --grpc on windows
    # also raising warning if users running on MacOS or FreeBSD
    if psutil.WINDOWS and (not development_mode):
        raise BentoMLException(
            "'grpc' is not supported on Windows without '--development'. The reason being SO_REUSEPORT socket option is only available on UNIX system, and gRPC implementation depends on this behaviour."
        )
    if psutil.MACOS or psutil.FREEBSD:
        logger.warning(
            "Due to gRPC implementation on exposing SO_REUSEPORT, BentoML production server's behaviour on %s is not correct. We recommend to containerize BentoServer as a Linux container instead. For testing locally, use `bentoml serve --development`",
            "MacOS" if psutil.MACOS else "FreeBSD",
        )

    # NOTE: We need to find and set model-repository args
    # to all TritonRunner instances (required from tritonserver if spawning multiple instances.)

    if psutil.POSIX and not IS_WSL:
        # use AF_UNIX sockets for Circus
        uds_path = tempfile.mkdtemp()
        get_socket_func = _get_runner_socket_posix
    elif psutil.WINDOWS or IS_WSL:
        get_socket_func = _get_runner_socket_windows
    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")
    with contextlib.ExitStack() as port_stack:
        for runner in svc.runners:
            if isinstance(runner, Runner):
                if runner.embedded or development_mode:
                    continue

                socket_uri, circus_socket = get_socket_func(
                    runner, uds_path, port_stack, backlog
                )

                runner_bind_map[runner.name] = socket_uri
                circus_socket_map[runner.name] = circus_socket

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
                        env=env,
                    )
                )
            else:
                # Make sure that the tritonserver uses the correct protocol
                runner_bind_map[runner.name] = runner.protocol_address
                cli_args = runner.cli_args + [
                    (
                        f"--http-port={runner.protocol_address.split(':')[-1]}"
                        if runner.tritonserver_type == "http"
                        else f"--grpc-port={runner.protocol_address.split(':')[-1]}"
                    )
                ]
                watchers.append(
                    create_watcher(
                        name=f"tritonserver_{runner.name}",
                        cmd=find_triton_binary(),
                        args=cli_args,
                        use_sockets=False,
                        working_dir=working_dir,
                        numprocesses=1,
                        env=env,
                    )
                )
        # reserve one more to avoid conflicts
        port_stack.enter_context(reserve_free_port())

    logger.debug("Runner map: %s", runner_bind_map)

    ssl_args = construct_ssl_args(
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_ca_certs=ssl_ca_certs,
    )
    scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"

    close_child_stdin: bool = False if development_mode else True

    on_service_deployment(svc)

    with contextlib.ExitStack() as port_stack:
        api_port = port_stack.enter_context(
            reserve_free_port(host, port=port, enable_so_reuseport=True)
        )
        api_server_args = [
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
            "--worker-id",
            "$(CIRCUS.WID)",
            *ssl_args,
            "--protocol-version",
            protocol_version,
        ]

        if reflection:
            api_server_args.append("--enable-reflection")
        if channelz:
            api_server_args.append("--enable-channelz")
        if max_concurrent_streams:
            api_server_args.extend(
                [
                    "--max-concurrent-streams",
                    str(max_concurrent_streams),
                ]
            )

        if development_mode:
            api_server_args.append("--development-mode")

        watchers.append(
            create_watcher(
                name="grpc_api_server",
                args=api_server_args,
                use_sockets=False,
                working_dir=working_dir,
                numprocesses=api_workers,
                close_child_stdin=close_child_stdin,
                env=env,
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
                    "--backlog",
                    f"{backlog}",
                ],
                working_dir=working_dir,
                numprocesses=1,
                singleton=True,
                close_child_stdin=close_child_stdin,
            )
        )

        log_metrics_host = "127.0.0.1" if metrics_host == "0.0.0.0" else metrics_host

        logger.info(
            PROMETHEUS_MESSAGE,
            "gRPC",
            bento_identifier,
            f"http://{log_metrics_host}:{metrics_port}",
        )

    arbiter_kwargs: dict[str, t.Any] = {
        "watchers": watchers,
        "sockets": list(circus_socket_map.values()),
        "threaded": threaded,
    }

    plugins = []

    if reload:
        reload_plugin = make_reload_plugin(working_dir, bentoml_home)
        plugins.append(reload_plugin)

    arbiter_kwargs["plugins"] = plugins

    if development_mode:
        arbiter_kwargs["debug"] = True if sys.platform != "win32" else False
        arbiter_kwargs["loggerconfig"] = SERVER_LOGGING_CONFIG
        arbiter_kwargs["loglevel"] = "WARNING"

    arbiter = create_standalone_arbiter(**arbiter_kwargs)

    production: bool = False if development_mode else True

    arbiter.exit_stack.enter_context(
        track_serve(svc, production=production, serve_kind="grpc")
    )

    @arbiter.exit_stack.callback
    def cleanup():
        if uds_path is not None:
            shutil.rmtree(uds_path)

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
        return Server(url=f"{scheme}://{host}:{port}", arbiter=arbiter)
    except Exception:
        cleanup()
        raise
