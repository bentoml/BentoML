from __future__ import annotations

import os
import json
import logging
import contextlib

from simple_di import inject
from simple_di import Provide

from .grpc.utils import LATEST_PROTOCOL_VERSION
from ._internal.runner.runner import Runner
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
    from .serve import ensure_prometheus_dir

    prometheus_dir = ensure_prometheus_dir()

    from . import load
    from .serve import create_watcher
    from .serve import find_triton_binary
    from ._internal.utils import reserve_free_port
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: list[Watcher] = []
    circus_socket_map: dict[str, CircusSocket] = {}

    # NOTE: We need to find and set model-repository args
    # to all TritonRunner instances (required from tritonserver if spawning multiple instances.)

    with contextlib.ExitStack() as port_stack:
        for runner in svc.runners:
            if runner.name == runner_name:
                if isinstance(runner, Runner):
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
                        create_watcher(
                            name=f"{RUNNER}_{runner.name}",
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
                                "--prometheus-dir",
                                prometheus_dir,
                            ],
                            working_dir=working_dir,
                            numprocesses=runner.scheduled_worker_count,
                        )
                    )
                    break
                else:
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
    runner_map: dict[str, str],
    working_dir: str,
    port: int = Provide[BentoMLContainer.api_server_config.port],
    host: str = Provide[BentoMLContainer.api_server_config.host],
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
    from .serve import ensure_prometheus_dir

    prometheus_dir = ensure_prometheus_dir()

    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

    from . import load
    from .serve import create_watcher
    from .serve import API_SERVER_NAME
    from .serve import construct_ssl_args
    from .serve import PROMETHEUS_MESSAGE
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)
    runner_requirements = {runner.name for runner in svc.runners}
    if not runner_requirements.issubset(set(runner_map)):
        raise ValueError(
            f"{bento_identifier} requires runners {runner_requirements}, but only {set(runner_map)} are provided."
        )
    watchers: list[Watcher] = []
    circus_socket_map: dict[str, CircusSocket] = {}
    logger.debug("Runner map: %s", runner_map)
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
                json.dumps(runner_map),
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
        logger.info(
            PROMETHEUS_MESSAGE,
            scheme.upper(),
            bento_identifier,
            f"{scheme}://{host}:{port}/metrics",
        )

    arbiter = create_standalone_arbiter(
        watchers=watchers,
        sockets=list(circus_socket_map.values()),
    )
    with track_serve(svc, production=True, component=API_SERVER):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting bare %s BentoServer from "%s" running on %s://%s:%d (Press CTRL+C to quit)',
                scheme.upper(),
                bento_identifier,
                scheme,
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
    api_workers: int = Provide[BentoMLContainer.api_server_workers],
    reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
    channelz: bool = Provide[BentoMLContainer.grpc.channelz.enabled],
    max_concurrent_streams: int
    | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    protocol_version: str = LATEST_PROTOCOL_VERSION,
) -> None:
    from .serve import ensure_prometheus_dir

    prometheus_dir = ensure_prometheus_dir()

    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

    from bentoml import load

    from .serve import create_watcher
    from .serve import construct_ssl_args
    from .serve import PROMETHEUS_MESSAGE
    from .serve import PROMETHEUS_SERVER_NAME
    from ._internal.utils import reserve_free_port
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)
    runner_requirements = {runner.name for runner in svc.runners}
    if not runner_requirements.issubset(set(runner_map)):
        raise ValueError(
            f"{bento_identifier} requires runners {runner_requirements}, but only {set(runner_map)} are provided."
        )
    watchers: list[Watcher] = []
    circus_socket_map: dict[str, CircusSocket] = {}
    logger.debug("Runner map: %s", runner_map)
    ssl_args = construct_ssl_args(
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_ca_certs=ssl_ca_certs,
    )
    scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"
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
                'Starting bare %s BentoServer from "%s" running on %s://%s:%d (Press CTRL+C to quit)',
                "gRPC",
                bento_identifier,
                scheme,
                host,
                port,
            ),
        )
