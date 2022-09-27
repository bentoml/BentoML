from __future__ import annotations

import os
import sys
import json
import math
import typing as t
import logging
import contextlib

from simple_di import inject
from simple_di import Provide

from ._internal.configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

SCRIPT_RUNNER = "bentoml_cli.worker.runner"
SCRIPT_API_SERVER = "bentoml_cli.worker.http_api_server"
SCRIPT_GRPC_API_SERVER = "bentoml_cli.worker.grpc_api_server"
SCRIPT_GRPC_PROMETHEUS_SERVER = "bentoml_cli.worker.grpc_prometheus_server"

API_SERVER = "api_server"
RUNNER = "runner"


@inject
def start_runner_server(
    bento_identifier: str,
    working_dir: str,
    runner_name: str,
    port: int | None = None,
    host: str | None = None,
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
) -> None:
    """
    Experimental API for serving a BentoML runner.
    """
    from bentoml import load

    from .serve import ensure_prometheus_dir
    from ._internal.utils import reserve_free_port
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}

    ensure_prometheus_dir()

    with contextlib.ExitStack() as port_stack:
        for runner in svc.runners:
            if runner.name == runner_name:
                if port is None:
                    port = port_stack.enter_context(reserve_free_port())
                if host is None:
                    host = "127.0.0.1"
                circus_socket_map[runner.name] = CircusSocket(
                    name=runner.name,
                    host=host,
                    port=port,
                    backlog=backlog,
                )

                watchers.append(
                    Watcher(
                        name=f"{RUNNER}_{runner.name}",
                        cmd=sys.executable,
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
                        ],
                        copy_env=True,
                        stop_children=True,
                        use_sockets=True,
                        working_dir=working_dir,
                        numprocesses=runner.scheduled_worker_count,
                    )
                )
                break
        else:
            raise ValueError(
                f"Runner {runner_name} not found in the service: `{bento_identifier}`, "
                f"available runners: {[r.name for r in svc.runners]}"
            )
    arbiter = create_standalone_arbiter(
        watchers=watchers,
        sockets=list(circus_socket_map.values()),
    )
    with track_serve(svc, production=True, component=RUNNER):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting RunnerServer from "%s" running on http://%s:%s (Press CTRL+C to quit)',
                bento_identifier,
                host,
                port,
            ),
        )


@inject
def start_http_server(
    bento_identifier: str,
    runner_map: t.Dict[str, str],
    working_dir: str,
    port: int = Provide[BentoMLContainer.api_server_config.port],
    host: str = Provide[BentoMLContainer.api_server_config.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
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
) -> None:
    from bentoml import load

    from .serve import create_watcher
    from .serve import API_SERVER_NAME
    from .serve import construct_ssl_args
    from .serve import PROMETHEUS_MESSAGE
    from .serve import ensure_prometheus_dir
    from ._internal.resource import CpuResource
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    runner_requirements = {runner.name for runner in svc.runners}
    if not runner_requirements.issubset(set(runner_map)):
        raise ValueError(
            f"{bento_identifier} requires runners {runner_requirements}, but only {set(runner_map)} are provided."
        )

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}

    prometheus_dir = ensure_prometheus_dir()

    logger.debug("Runner map: %s", runner_map)

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
                "--fd",
                f"$(circus.sockets.{API_SERVER_NAME})",
                "--runner-map",
                json.dumps(runner_map),
                "--working-dir",
                working_dir,
                "--backlog",
                f"{backlog}",
                "--worker-id",
                "$(CIRCUS.WID)",
                "--prometheus-dir",
                prometheus_dir,
                *construct_ssl_args(
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
    if BentoMLContainer.api_server_config.metrics.enabled.get():
        logger.info(
            PROMETHEUS_MESSAGE,
            "HTTP",
            bento_identifier,
            f"http://{host}:{port}/metrics",
        )

    arbiter = create_standalone_arbiter(
        watchers=watchers,
        sockets=list(circus_socket_map.values()),
    )
    with track_serve(svc, production=True, component=API_SERVER):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting bare %s BentoServer from "%s" running on http://%s:%d (Press CTRL+C to quit)',
                "HTTP",
                bento_identifier,
                host,
                port,
            ),
        )


@inject
def start_grpc_server(
    bento_identifier: str,
    runner_map: dict[str, str],
    working_dir: str,
    port: int = Provide[BentoMLContainer.grpc.port],
    host: str = Provide[BentoMLContainer.grpc.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    api_workers: int | None = None,
    reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
    max_concurrent_streams: int
    | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
) -> None:
    from bentoml import load

    from .serve import create_watcher
    from .serve import PROMETHEUS_MESSAGE
    from .serve import ensure_prometheus_dir
    from .serve import PROMETHEUS_SERVER_NAME
    from ._internal.utils import reserve_free_port
    from ._internal.resource import CpuResource
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    runner_requirements = {runner.name for runner in svc.runners}
    if not runner_requirements.issubset(set(runner_map)):
        raise ValueError(
            f"{bento_identifier} requires runners {runner_requirements}, but only {set(runner_map)} are provided."
        )

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: list[Watcher] = []
    circus_socket_map: dict[str, CircusSocket] = {}
    prometheus_dir = ensure_prometheus_dir()
    logger.debug("Runner map: %s", runner_map)
    with contextlib.ExitStack() as port_stack:
        api_port = port_stack.enter_context(
            reserve_free_port(host=host, port=port, enable_so_reuseport=True)
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
            json.dumps(runner_map),
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

        logger.info(
            PROMETHEUS_MESSAGE,
            "gRPC",
            bento_identifier,
            f"http://{metrics_host}:{metrics_port}",
        )
    arbiter = create_standalone_arbiter(
        watchers=watchers, sockets=list(circus_socket_map.values())
    )
    with track_serve(svc, production=True, component=API_SERVER, serve_kind="grpc"):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting bare %s BentoServer from "%s" running on http://%s:%d (Press CTRL+C to quit)',
                "gRPC",
                bento_identifier,
                host,
                port,
            ),
        )
