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

from simple_di import inject
from simple_di import Provide

from bentoml import load

from ._internal.utils import reserve_free_port
from ._internal.resource import CpuResource
from ._internal.utils.circus import create_standalone_arbiter
from ._internal.configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

SCRIPT_RUNNER = "bentoml_cli.worker.runner"
SCRIPT_API_SERVER = "bentoml_cli.worker.http_api_server"
SCRIPT_DEV_API_SERVER = "bentoml_cli.worker.http_dev_api_server"


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


MAX_AF_UNIX_PATH_LENGTH = 103


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
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}
    uds_path = None

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
                        name=f"runner_{runner.name}",
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

    try:
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting RunnerServer from "%s"\n running on http://%s:%s (Press CTRL+C to quit)',
                bento_identifier,
                host,
                port,
            ),
        )
    finally:
        if uds_path is not None:
            shutil.rmtree(uds_path)


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
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    runner_requirements = {runner.name for runner in svc.runners}
    if not runner_requirements.issubset(set(runner_map)):
        raise ValueError(
            f"{bento_identifier} requires runners {runner_requirements}, but only "
            f"{set(runner_map)} are provided"
        )

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}
    uds_path = None

    prometheus_dir = ensure_prometheus_dir()

    logger.debug("Runner map: %s", runner_map)

    circus_socket_map["_bento_api_server"] = CircusSocket(
        name="_bento_api_server",
        host=host,
        port=port,
        backlog=backlog,
    )

    args: list[str | int] = [
        "-m",
        SCRIPT_API_SERVER,
        bento_identifier,
        "--fd",
        "$(circus.sockets._bento_api_server)",
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
    ]

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
        args.extend(["--ssl-version", int(ssl_version)])
    if ssl_cert_reqs:
        args.extend(["--ssl-cert-reqs", int(ssl_cert_reqs)])
    if ssl_ciphers:
        args.extend(["--ssl-ciphers", ssl_ciphers])

    watchers.append(
        Watcher(
            name="api_server",
            cmd=sys.executable,
            args=args,
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

    try:
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                f'Starting bare Bento API server from "{bento_identifier}" '
                f"running on http://{host}:{port} (Press CTRL+C to quit)"
            ),
        )
    finally:
        if uds_path is not None:
            shutil.rmtree(uds_path)
