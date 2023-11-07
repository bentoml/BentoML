from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import os
import platform
import shutil
import socket
import tempfile
import typing as t

from simple_di import Provide
from simple_di import inject

from bentoml._internal.container import BentoMLContainer
from bentoml.exceptions import BentoMLConfigException

from .service import Service

if t.TYPE_CHECKING:
    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

POSIX = os.name == "posix"
WINDOWS = os.name == "nt"
IS_WSL = "microsoft-standard" in platform.release()
API_SERVER_NAME = "_bento_api_server"
PROMETHEUS_MESSAGE = (
    'Prometheus metrics for %s BentoServer from "%s" can be accessed at %s.'
)

MAX_AF_UNIX_PATH_LENGTH = 103
logger = logging.getLogger("bentoml.serve")

if POSIX and not IS_WSL:

    def _get_server_socket(
        service: Service,
        uds_path: str | None,
        port_stack: contextlib.ExitStack,
        backlog: int,
    ) -> tuple[str, CircusSocket]:
        from circus.sockets import CircusSocket

        from bentoml._internal.utils.uri import path_to_uri

        assert uds_path is not None
        socket_path = os.path.join(uds_path, f"{id(service)}.sock")
        assert len(socket_path) < MAX_AF_UNIX_PATH_LENGTH
        return path_to_uri(socket_path), CircusSocket(
            name=service.name, path=socket_path, backlog=backlog
        )

elif WINDOWS or IS_WSL:

    def _get_server_socket(
        service: Service,
        uds_path: str | None,
        port_stack: contextlib.ExitStack,
        backlog: int,
    ) -> tuple[str, CircusSocket]:
        from circus.sockets import CircusSocket

        from bentoml._internal.utils import reserve_free_port

        runner_port = port_stack.enter_context(reserve_free_port())
        runner_host = "127.0.0.1"

        return f"tcp://{runner_host}:{runner_port}", CircusSocket(
            name=service.name,
            host=runner_host,
            port=runner_port,
            backlog=backlog,
        )

else:

    def _get_server_socket(
        service: Service,
        uds_path: str | None,
        port_stack: contextlib.ExitStack,
        backlog: int,
    ) -> tuple[str, CircusSocket]:
        from bentoml.exceptions import BentoMLException

        raise BentoMLException("Unsupported platform")


SERVICE_WORKER_SCRIPT = "bentoml_io.worker.service"


def create_service_watchers(
    svc: Service,
    work_dir: str,
    uds_path: str | None,
    port_stack: contextlib.ExitStack,
    backlog: int,
    dependency_map: dict[str, str],
) -> tuple[list[Watcher], list[CircusSocket], str]:
    from bentoml.serve import create_watcher

    watchers: list[Watcher] = []
    sockets: list[CircusSocket] = []
    for dep in svc.dependencies:
        if dep.name in dependency_map:
            continue
        new_watchers, new_sockets, uri = create_service_watchers(
            dep, work_dir, uds_path, port_stack, backlog, dependency_map
        )
        watchers.extend(new_watchers)
        sockets.extend(new_sockets)
        dependency_map[dep.name] = uri

    uri, socket = _get_server_socket(svc, uds_path, port_stack, backlog)
    sockets.append(socket)
    args = [
        "-m",
        SERVICE_WORKER_SCRIPT,
        svc.import_string,
        "--fd",
        f"$(circus.sockets.{svc.name})",
        "--working-dir",
        work_dir,
        "--worker-id",
        "$(CIRCUS.WID)",
    ]

    if env_map := svc.worker_env_map:
        args.extend(["--worker-env", json.dumps(env_map)])

    watchers.append(
        create_watcher(
            name=f"dependency_{svc.name}",
            args=args,
            working_dir=work_dir,
            numprocesses=svc.worker_count,
        )
    )

    return watchers, sockets, uri


@inject
def serve_http_production(
    bento_identifier: str | Service,
    working_dir: str,
    host: str = Provide[BentoMLContainer.http.host],
    port: int = Provide[BentoMLContainer.http.port],
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
) -> None:
    from circus.sockets import CircusSocket

    from bentoml._internal.log import SERVER_LOGGING_CONFIG
    from bentoml._internal.service.loader import load
    from bentoml._internal.utils import reserve_free_port
    from bentoml._internal.utils.circus import create_standalone_arbiter
    from bentoml.serve import construct_ssl_args
    from bentoml.serve import create_watcher
    from bentoml.serve import ensure_prometheus_dir
    from bentoml.serve import make_reload_plugin

    prometheus_dir = ensure_prometheus_dir()
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    if isinstance(bento_identifier, Service):
        svc = bento_identifier
    else:
        svc = t.cast("Service", load(bento_identifier, working_dir))
    BentoMLContainer.api_server_workers.set(api_workers)

    watchers: list[Watcher] = []
    sockets: list[CircusSocket] = []
    dependency_map: dict[str, str] = {}
    with contextlib.ExitStack() as stack:
        uds_path: str | None = None
        if POSIX and not IS_WSL:
            uds_path = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="bentoml-uds-")
            )
        if not development_mode:
            for dep in svc.dependencies:
                if dep.name in dependency_map:
                    continue
                new_watchers, new_sockets, uri = create_service_watchers(
                    dep, working_dir, uds_path, stack, backlog, dependency_map
                )
                watchers.extend(new_watchers)
                sockets.extend(new_sockets)
                dependency_map[dep.name] = uri
            # reserve one more to avoid conflicts
            stack.enter_context(reserve_free_port())

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

        sockets.append(
            CircusSocket(
                name=API_SERVER_NAME,
                host=host,
                port=port,
                family=family,
                backlog=backlog,
            )
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
        timeout_args = ["--timeout", str(timeout)] if timeout else []

        server_args = [
            "-m",
            SERVICE_WORKER_SCRIPT,
            bento_identifier
            if isinstance(bento_identifier, str)
            else bento_identifier.import_string,
            "--fd",
            f"$(circus.sockets.{API_SERVER_NAME})",
            "--working-dir",
            working_dir,
            "--backlog",
            str(backlog),
            "--worker-id",
            "$(CIRCUS.WID)",
            "--prometheus-dir",
            prometheus_dir,
            *ssl_args,
            *timeout_args,
        ]
        if env_map := svc.worker_env_map:
            server_args.extend(["--worker-env", json.dumps(env_map)])
        if development_mode:
            server_args.append("--development-mode")

        scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"
        watchers.append(
            create_watcher(
                name="api_server",
                args=server_args,
                working_dir=working_dir,
                numprocesses=svc.worker_count,
                close_child_stdin=not development_mode,
            )
        )
        if BentoMLContainer.api_server_config.metrics.enabled.get():
            log_host = "localhost" if host in ["0.0.0.0", "::"] else host

            logger.info(
                PROMETHEUS_MESSAGE,
                scheme.upper(),
                bento_identifier,
                f"{scheme}://{log_host}:{port}/metrics",
            )

        # inject runner map now
        inject_env = {"BENTOML_RUNNER_MAP": json.dumps(dependency_map)}
        for watcher in watchers:
            if watcher.env is None:
                watcher.env = inject_env
            else:
                watcher.env.update(inject_env)

        arbiter_kwargs: dict[str, t.Any] = {"watchers": watchers, "sockets": sockets}

        if reload:
            reload_plugin = make_reload_plugin(working_dir, bentoml_home)
            arbiter_kwargs["plugins"] = [reload_plugin]

        if development_mode:
            arbiter_kwargs["debug"] = True
            arbiter_kwargs["loggerconfig"] = SERVER_LOGGING_CONFIG
            arbiter_kwargs["loglevel"] = "WARNING"

        arbiter = create_standalone_arbiter(**arbiter_kwargs)

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
