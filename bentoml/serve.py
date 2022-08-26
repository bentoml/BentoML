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

from ._internal.log import SERVER_LOGGING_CONFIG
from ._internal.utils import reserve_free_port
from ._internal.resource import CpuResource
from ._internal.utils.uri import path_to_uri
from ._internal.utils.circus import create_standalone_arbiter
from ._internal.utils.analytics import track_serve
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


@inject
def serve_development(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoMLContainer.api_server_config.port],
    host: str = Provide[BentoMLContainer.api_server_config.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
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
) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(bento_identifier, working_dir=working_dir)  # verify service loading

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    prometheus_dir = ensure_prometheus_dir()

    watchers: t.List[Watcher] = []

    circus_sockets: t.List[CircusSocket] = []
    circus_sockets.append(
        CircusSocket(
            name="_bento_api_server",
            host=host,
            port=port,
            backlog=backlog,
        )
    )

    args: list[str | int] = [
        "-m",
        SCRIPT_DEV_API_SERVER,
        bento_identifier,
        "--fd",
        "$(circus.sockets._bento_api_server)",
        "--working-dir",
        working_dir,
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
            name="dev_api_server",
            cmd=sys.executable,
            args=args,
            copy_env=True,
            stop_children=True,
            use_sockets=True,
            working_dir=working_dir,
            # we don't want to close stdin for child process in case user use debugger.
            # See https://circus.readthedocs.io/en/latest/for-ops/configuration/
            close_child_stdin=False,
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
        debug=True if sys.platform != "win32" else False,
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


MAX_AF_UNIX_PATH_LENGTH = 103


@inject
def serve_production(
    bento_identifier: str,
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

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}
    runner_bind_map: t.Dict[str, str] = {}
    uds_path = None

    prometheus_dir = ensure_prometheus_dir()

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
                        "--fd",
                        f"$(circus.sockets.{runner.name})",
                        "--working-dir",
                        working_dir,
                        "--worker-id",
                        "$(CIRCUS.WID)",
                        "--worker-env-map",
                        json.dumps(runner.scheduled_worker_env_map),
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
                            "--fd",
                            f"$(circus.sockets.{runner.name})",
                            "--working-dir",
                            working_dir,
                            "--no-access-log",
                            "--worker-id",
                            "$(circus.wid)",
                            "--worker-env-map",
                            json.dumps(runner.scheduled_worker_env_map),
                        ],
                        copy_env=True,
                        stop_children=True,
                        use_sockets=True,
                        working_dir=working_dir,
                        numprocesses=runner.scheduled_worker_count,
                    )
                )
            port_stack.enter_context(
                reserve_free_port()
            )  # reserve one more to avoid conflicts
    else:
        raise NotImplementedError("Unsupported platform: {}".format(sys.platform))

    logger.debug("Runner map: %s", runner_bind_map)

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
        json.dumps(runner_bind_map),
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
