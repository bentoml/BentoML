from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import os
import pathlib
import platform
import shutil
import socket
import sys
import tempfile
import typing as t

from simple_di import Provide
from simple_di import inject

from _bentoml_sdk import Service
from bentoml._internal.container import BentoMLContainer
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils.circus import Server
from bentoml.exceptions import BentoMLConfigException

AnyService = Service[t.Any]

if t.TYPE_CHECKING:
    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

    from .allocator import ResourceAllocator

POSIX = os.name == "posix"
WINDOWS = os.name == "nt"
IS_WSL = "microsoft-standard" in platform.release()
API_SERVER_NAME = "_bento_api_server"

MAX_AF_UNIX_PATH_LENGTH = 103
logger = logging.getLogger("bentoml.serve")


def _get_server_socket(
    service: AnyService,
    uds_path: str,
    port_stack: contextlib.ExitStack,
    backlog: int,
) -> tuple[str, CircusSocket]:
    from circus.sockets import CircusSocket

    from bentoml._internal.utils.uri import path_to_uri

    if WINDOWS or IS_WSL or "BENTOML_NO_UDS" in os.environ:
        runner_port = port_stack.enter_context(reserve_free_port())
        runner_host = "127.0.0.1"

        return f"tcp://{runner_host}:{runner_port}", CircusSocket(
            name=service.name,
            host=runner_host,
            port=runner_port,
            backlog=backlog,
        )

    socket_path = os.path.join(uds_path, f"{id(service)}.sock")
    assert len(socket_path) < MAX_AF_UNIX_PATH_LENGTH
    return path_to_uri(socket_path), CircusSocket(
        name=service.name, path=socket_path, backlog=backlog
    )


_SERVICE_WORKER_SCRIPT = "_bentoml_impl.worker.service"


@inject
def create_dependency_watcher(
    bento_identifier: str,
    svc: AnyService,
    uds_path: str,
    port_stack: contextlib.ExitStack,
    backlog: int,
    scheduler: ResourceAllocator,
    working_dir: str | None = None,
    env: dict[str, str] | None = None,
    bento_args: dict[str, t.Any] = Provide[BentoMLContainer.bento_arguments],
) -> tuple[Watcher, CircusSocket | None, str]:
    from bentoml.exceptions import BentoMLException
    from bentoml.serving import create_watcher

    num_workers, worker_envs = scheduler.get_worker_env(svc)
    env = env or {}
    uri, socket = _get_server_socket(svc, uds_path, port_stack, backlog)
    args = [
        "-m",
        _SERVICE_WORKER_SCRIPT,
        bento_identifier,
        "--service-name",
        svc.name,
        "--fd",
        f"$(circus.sockets.{svc.name})",
        "--worker-id",
        "$(CIRCUS.WID)",
        "--args",
        json.dumps(bento_args),
    ]
    cmd = sys.executable

    # Process environment variables from the service
    for env_var in svc.envs:
        # Exclude Build Time environment variables
        if env_var.stage == "build":
            continue
        if env_var.name in env:
            continue

        if env_var.name in os.environ:
            env[env_var.name] = os.environ[env_var.name]
        elif env_var.value:
            env[env_var.name] = env_var.value
        else:
            raise BentoMLException(
                f"Environment variable '{env_var.name}' is required but not set. "
                f"Either set it in the environment or provide a default value in the service definition."
            )

    env.update(worker_envs)
    watcher = create_watcher(
        name=f"service_{svc.name}",
        cmd=cmd,
        args=args,
        numprocesses=num_workers,
        working_dir=working_dir,
        env=env,
    )
    return watcher, socket, uri


@inject
def server_on_deployment(
    svc: AnyService, result_file: str = Provide[BentoMLContainer.result_store_file]
) -> None:
    # Resolve models before server starts.
    from ..tasks.result import Sqlite3Store

    if bento := svc.bento:
        for model in bento.info.all_models:
            model.to_model().resolve()
    else:
        for model in svc.models:
            model.resolve()
    for name in dir(svc.inner):
        member = getattr(svc.inner, name)
        if callable(member) and getattr(member, "__bentoml_deployment_hook__", False):
            member()
    if svc.needs_task_db():
        if os.path.exists(result_file):
            os.remove(result_file)
        Sqlite3Store.init_db(result_file)


@inject(squeeze_none=True)
def serve_http(
    bento_identifier: str | AnyService,
    working_dir: str | None = None,
    host: str = Provide[BentoMLContainer.http.host],
    port: int = Provide[BentoMLContainer.http.port],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
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
    dependency_map: dict[str, str] | None = None,
    service_name: str = "",
    threaded: bool = False,
) -> Server:
    from circus.sockets import CircusSocket

    from bentoml._internal.log import SERVER_LOGGING_CONFIG
    from bentoml._internal.utils import reserve_free_port
    from bentoml._internal.utils.analytics.usage_stats import track_serve
    from bentoml._internal.utils.circus import create_standalone_arbiter
    from bentoml.exceptions import BentoMLException
    from bentoml.serving import construct_ssl_args
    from bentoml.serving import construct_timeouts_args
    from bentoml.serving import create_watcher
    from bentoml.serving import ensure_prometheus_dir
    from bentoml.serving import make_reload_plugin

    from ..loader import load
    from .allocator import ResourceAllocator

    env = {"PROMETHEUS_MULTIPROC_DIR": ensure_prometheus_dir()}
    if isinstance(bento_identifier, Service):
        svc = bento_identifier
        assert working_dir is None, (
            "working_dir should not be set when passing a service in process"
        )
        bento_identifier = svc.import_string
        bento_path = pathlib.Path(svc.working_dir)
    else:
        svc = load(bento_identifier, working_dir)
        bento_path = pathlib.Path(working_dir or ".")

    # Process environment variables from the service
    for env_var in svc.envs:
        # Exclude Build Time environment variables
        if env_var.stage == "build":
            continue
        if env_var.name in env:
            continue

        if env_var.name in os.environ:
            env[env_var.name] = os.environ[env_var.name]
        elif env_var.value:
            env[env_var.name] = env_var.value
        else:
            raise BentoMLException(
                f"Environment variable '{env_var.name}' is required but not set. "
                f"Either set it in the environment or provide a default value in the service definition."
            )

    watchers: list[Watcher] = []
    sockets: list[CircusSocket] = []
    allocator = ResourceAllocator()
    if dependency_map is None:
        dependency_map = {}
    if service_name and service_name != svc.name:
        svc = svc.find_dependent_by_name(service_name)
    num_workers, worker_env = allocator.get_worker_env(svc)
    server_on_deployment(svc)
    uds_path = tempfile.mkdtemp(prefix="bentoml-uds-")
    try:
        if not service_name and not development_mode:
            with contextlib.ExitStack() as port_stack:
                for name, dep_svc in svc.all_services(exclude_urls=True).items():
                    if name == svc.name or name in dependency_map:
                        continue

                    # Process environment variables for dependency services
                    dependency_env = env.copy()
                    for env_var in dep_svc.envs:
                        # Exclude Build Time environment variables
                        if env_var.stage == "build":
                            continue
                        if env_var.name in dependency_env:
                            continue

                        if env_var.value:
                            dependency_env[env_var.name] = env_var.value
                        elif env_var.name in os.environ:
                            dependency_env[env_var.name] = os.environ[env_var.name]
                        else:
                            raise BentoMLException(
                                f"Environment variable '{env_var.name}' is required for service '{name}' but not set. "
                                f"Either set it in the environment or provide a default value in the service definition."
                            )

                    new_watcher, new_socket, uri = create_dependency_watcher(
                        bento_identifier,
                        dep_svc,
                        uds_path,
                        port_stack,
                        backlog,
                        allocator,
                        str(bento_path.absolute()),
                        env={k: str(v) for k, v in dependency_env.items()},
                    )
                    watchers.append(new_watcher)
                    if new_socket:
                        sockets.append(new_socket)
                    dependency_map[name] = uri
                    server_on_deployment(dep_svc)
                # reserve one more to avoid conflicts
                port_stack.enter_context(reserve_free_port())

        try:
            ipaddr = ipaddress.ip_address(host)
            if ipaddr.version == 4:
                family = socket.AF_INET
            elif ipaddr.version == 6:
                family = socket.AF_INET6
            else:
                raise BentoMLConfigException(
                    f"Unsupported host IP address version: {ipaddr.version}"
                )
        except ValueError as e:
            raise BentoMLConfigException(f"Invalid host IP address: {host}") from e
        bento_args = BentoMLContainer.bento_arguments.get()
        env.update(worker_env)
        sockets.append(
            CircusSocket(
                name=API_SERVER_NAME,
                host=host,
                port=port,
                family=family,
                backlog=backlog,
            )
        )
        if BentoMLContainer.ssl.enabled.get() and not ssl_certfile:
            raise BentoMLConfigException("ssl_certfile is required when ssl is enabled")

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
        timeout_args = ["--timeout", str(timeout)] if timeout else []

        server_cmd = sys.executable
        server_args = [
            "-m",
            _SERVICE_WORKER_SCRIPT,
            bento_identifier,
            "--fd",
            f"$(circus.sockets.{API_SERVER_NAME})",
            "--service-name",
            svc.name,
            "--backlog",
            str(backlog),
            "--worker-id",
            "$(CIRCUS.WID)",
            "--args",
            json.dumps(bento_args),
            *ssl_args,
            *timeouts_args,
            *timeout_args,
        ]
        if development_mode:
            server_args.append("--development-mode")

        scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"
        watchers.append(
            create_watcher(
                name="service",
                cmd=server_cmd,
                args=server_args,
                working_dir=str(bento_path.absolute()),
                numprocesses=num_workers,
                close_child_stdin=not development_mode,
                env={k: str(v) for k, v in env.items()},
            )
        )

        log_host = "localhost" if host in ["0.0.0.0", "::"] else host
        dependency_map[svc.name] = f"{scheme}://{log_host}:{port}"

        # inject runner map now
        inject_env = {"BENTOML_RUNNER_MAP": json.dumps(dependency_map)}
        for watcher in watchers:
            if watcher.env is None:
                watcher.env = inject_env
            else:
                watcher.env.update(inject_env)

        arbiter_kwargs: dict[str, t.Any] = {
            "watchers": watchers,
            "sockets": sockets,
            "threaded": threaded,
        }

        if reload:
            reload_plugin = make_reload_plugin(str(bento_path.absolute()), bentoml_home)
            arbiter_kwargs["plugins"] = [reload_plugin]

        if development_mode:
            arbiter_kwargs["debug"] = True
            arbiter_kwargs["loggerconfig"] = SERVER_LOGGING_CONFIG

        arbiter = create_standalone_arbiter(**arbiter_kwargs)
        arbiter.exit_stack.enter_context(
            track_serve(svc, production=not development_mode)
        )
        arbiter.exit_stack.callback(shutil.rmtree, uds_path, ignore_errors=True)
        arbiter.start(
            cb=lambda _: (
                logger.info(  # type: ignore
                    'Starting production %s BentoServer from "%s" listening on %s://%s:%d (Press CTRL+C to quit)',
                    scheme.upper(),
                    bento_identifier,
                    scheme,
                    log_host,
                    port,
                )
                if not svc.has_custom_command()
                else None
            ),
        )
        return Server(url=f"{scheme}://{log_host}:{port}", arbiter=arbiter)
    except Exception:
        shutil.rmtree(uds_path, ignore_errors=True)
        raise


_GRPC_SERVICE_WORKER_SCRIPT = "_bentoml_impl.worker.grpc_service"
_PROMETHEUS_SERVER_NAME = "_prometheus_server"


@inject(squeeze_none=True)
def serve_grpc(
    bento_identifier: str | AnyService,
    working_dir: str | None = None,
    host: str = Provide[BentoMLContainer.grpc.host],
    port: int = Provide[BentoMLContainer.grpc.port],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    development_mode: bool = False,
    reload: bool = False,
    dependency_map: dict[str, str] | None = None,
    service_name: str = "",
    threaded: bool = False,
    max_concurrent_streams: int | None = Provide[
        BentoMLContainer.grpc.max_concurrent_streams
    ],
    reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
    channelz: bool = Provide[BentoMLContainer.grpc.channelz.enabled],
    protocol_version: str = "v1",
) -> Server:
    import psutil
    from circus.sockets import CircusSocket

    from bentoml._internal.log import SERVER_LOGGING_CONFIG
    from bentoml._internal.utils import reserve_free_port
    from bentoml._internal.utils.analytics.usage_stats import track_serve
    from bentoml._internal.utils.circus import create_standalone_arbiter
    from bentoml.exceptions import BentoMLException
    from bentoml.serving import PROMETHEUS_MESSAGE
    from bentoml.serving import SCRIPT_GRPC_PROMETHEUS_SERVER
    from bentoml.serving import construct_ssl_args
    from bentoml.serving import create_watcher
    from bentoml.serving import ensure_prometheus_dir
    from bentoml.serving import make_reload_plugin

    from ..loader import load
    from .allocator import ResourceAllocator

    if protocol_version != "v1":
        raise BentoMLException(
            f"@bentoml.service() gRPC serving only supports protocol v1, got {protocol_version!r}"
        )

    if WINDOWS and not development_mode:
        raise BentoMLException(
            "'grpc' is not supported on Windows without '--development'. The reason being SO_REUSEPORT socket option is only available on UNIX system, and gRPC implementation depends on this behaviour."
        )
    if psutil.MACOS or psutil.FREEBSD:
        logger.warning(
            "Due to gRPC implementation on exposing SO_REUSEPORT, BentoML production server's behaviour on %s is not correct. We recommend to containerize BentoServer as a Linux container instead. For testing locally, use `bentoml serve-grpc --development`",
            "MacOS" if psutil.MACOS else "FreeBSD",
        )

    env = {"PROMETHEUS_MULTIPROC_DIR": ensure_prometheus_dir()}
    if isinstance(bento_identifier, Service):
        svc = bento_identifier
        assert working_dir is None, (
            "working_dir should not be set when passing a service in process"
        )
        bento_identifier = svc.import_string
        bento_path = pathlib.Path(svc.working_dir)
    else:
        svc = load(bento_identifier, working_dir)
        bento_path = pathlib.Path(working_dir or ".")

    for env_var in svc.envs:
        if env_var.stage == "build":
            continue
        if env_var.name in env:
            continue
        if env_var.name in os.environ:
            env[env_var.name] = os.environ[env_var.name]
        elif env_var.value:
            env[env_var.name] = env_var.value
        else:
            raise BentoMLException(
                f"Environment variable '{env_var.name}' is required but not set. "
                f"Either set it in the environment or provide a default value in the service definition."
            )

    watchers: list[Watcher] = []
    sockets: list[CircusSocket] = []
    allocator = ResourceAllocator()
    if dependency_map is None:
        dependency_map = {}
    if service_name and service_name != svc.name:
        svc = svc.find_dependent_by_name(service_name)
    num_workers, worker_env = allocator.get_worker_env(svc)
    if development_mode:
        num_workers = 1
    server_on_deployment(svc)
    uds_path = tempfile.mkdtemp(prefix="bentoml-uds-")
    try:
        if not service_name and not development_mode:
            with contextlib.ExitStack() as port_stack:
                for name, dep_svc in svc.all_services(exclude_urls=True).items():
                    if name == svc.name or name in dependency_map:
                        continue

                    dependency_env = env.copy()
                    for env_var in dep_svc.envs:
                        if env_var.stage == "build":
                            continue
                        if env_var.name in dependency_env:
                            continue
                        if env_var.value:
                            dependency_env[env_var.name] = env_var.value
                        elif env_var.name in os.environ:
                            dependency_env[env_var.name] = os.environ[env_var.name]
                        else:
                            raise BentoMLException(
                                f"Environment variable '{env_var.name}' is required for service '{name}' but not set. "
                                f"Either set it in the environment or provide a default value in the service definition."
                            )

                    new_watcher, new_socket, uri = create_dependency_watcher(
                        bento_identifier,
                        dep_svc,
                        uds_path,
                        port_stack,
                        backlog,
                        allocator,
                        str(bento_path.absolute()),
                        env={k: str(v) for k, v in dependency_env.items()},
                    )
                    watchers.append(new_watcher)
                    if new_socket:
                        sockets.append(new_socket)
                    dependency_map[name] = uri
                    server_on_deployment(dep_svc)
                port_stack.enter_context(reserve_free_port())

        env.update(worker_env)
        ssl_args = construct_ssl_args(
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            ssl_ca_certs=ssl_ca_certs,
        )
        scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"
        close_child_stdin = not development_mode
        bento_args = BentoMLContainer.bento_arguments.get()

        with contextlib.ExitStack() as port_stack:
            api_port = port_stack.enter_context(
                reserve_free_port(host, port=port, enable_so_reuseport=True)
            )
            server_args = [
                "-m",
                _GRPC_SERVICE_WORKER_SCRIPT,
                bento_identifier,
                "--host",
                host,
                "--port",
                str(api_port),
                "--working-dir",
                str(bento_path.absolute()),
                "--worker-id",
                "$(CIRCUS.WID)",
                "--args",
                json.dumps(bento_args),
                "--protocol-version",
                protocol_version,
                *ssl_args,
            ]
            if reflection:
                server_args.append("--enable-reflection")
            if channelz:
                server_args.append("--enable-channelz")
            if max_concurrent_streams:
                server_args.extend(
                    ["--max-concurrent-streams", str(max_concurrent_streams)]
                )
            if development_mode:
                server_args.append("--development-mode")

            watchers.append(
                create_watcher(
                    name="grpc_api_server",
                    args=server_args,
                    use_sockets=False,
                    working_dir=str(bento_path.absolute()),
                    numprocesses=num_workers,
                    close_child_stdin=close_child_stdin,
                    env={k: str(v) for k, v in env.items()},
                )
            )

        if BentoMLContainer.api_server_config.metrics.enabled.get():
            metrics_host = BentoMLContainer.grpc.metrics.host.get()
            metrics_port = BentoMLContainer.grpc.metrics.port.get()
            sockets.append(
                CircusSocket(
                    name=_PROMETHEUS_SERVER_NAME,
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
                        f"$(circus.sockets.{_PROMETHEUS_SERVER_NAME})",
                        "--backlog",
                        str(backlog),
                    ],
                    working_dir=str(bento_path.absolute()),
                    numprocesses=1,
                    singleton=True,
                    close_child_stdin=close_child_stdin,
                )
            )
            log_metrics_host = (
                "127.0.0.1" if metrics_host == "0.0.0.0" else metrics_host
            )
            logger.info(
                PROMETHEUS_MESSAGE,
                "gRPC",
                bento_identifier,
                f"http://{log_metrics_host}:{metrics_port}",
            )

        log_host = "localhost" if host in ["0.0.0.0", "::"] else host
        dependency_map[svc.name] = f"{scheme}://{log_host}:{port}"
        inject_env = {"BENTOML_RUNNER_MAP": json.dumps(dependency_map)}
        for watcher in watchers:
            if watcher.env is None:
                watcher.env = inject_env
            else:
                watcher.env.update(inject_env)

        arbiter_kwargs: dict[str, t.Any] = {
            "watchers": watchers,
            "sockets": sockets,
            "threaded": threaded,
        }
        if reload:
            arbiter_kwargs["plugins"] = [
                make_reload_plugin(str(bento_path.absolute()), bentoml_home)
            ]
        if development_mode:
            arbiter_kwargs["debug"] = True if sys.platform != "win32" else False
            arbiter_kwargs["loggerconfig"] = SERVER_LOGGING_CONFIG
            arbiter_kwargs["loglevel"] = "WARNING"

        arbiter = create_standalone_arbiter(**arbiter_kwargs)
        arbiter.exit_stack.enter_context(
            track_serve(svc, production=not development_mode, serve_kind="grpc")
        )
        arbiter.exit_stack.callback(shutil.rmtree, uds_path, ignore_errors=True)
        arbiter.start(
            cb=lambda _: (
                logger.info(  # type: ignore
                    'Starting production %s BentoServer from "%s" listening on %s://%s:%d (Press CTRL+C to quit)',
                    "gRPC",
                    bento_identifier,
                    scheme,
                    log_host,
                    port,
                )
                if not svc.has_custom_command()
                else None
            ),
        )
        return Server(url=f"{scheme}://{log_host}:{port}", arbiter=arbiter)
    except Exception:
        shutil.rmtree(uds_path, ignore_errors=True)
        raise
