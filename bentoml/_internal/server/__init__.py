import os
import sys
import time
import shutil
import typing as t
import logging
import tempfile
import contextlib
from urllib.parse import urlparse

import psutil
from simple_di import inject
from simple_di import Provide

from bentoml import load
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils.uri import path_to_uri
from bentoml._internal.utils.uri import uri_to_path
from bentoml._internal.utils.circus import create_standalone_arbiter

from ..configuration import get_debug_mode
from ..configuration.containers import DeploymentContainer

logger = logging.getLogger(__name__)

UVICORN_LOGGING_CONFIG: t.Dict[str, t.Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(message)s",
            "use_colors": False,
            "datefmt": "[%X]",
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(client_addr)s - "%(request_line)s" %(status_code)s',  # noqa: E501
            "use_colors": False,
            "datefmt": "[%X]",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "rich.logging.RichHandler",
        },
        "access": {
            "formatter": "access",
            "class": "rich.logging.RichHandler",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": [], "level": "INFO"},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}

SCRIPT_RUNNER = """
import bentoml._internal.server;
bentoml._internal.server.start_prod_runner_server("{bento_identifier}", "{runner_name}", working_dir="{working_dir}", instance_id=$(CIRCUS.WID), bind="{bind}");
"""

SCRIPT_API_SERVER = """
import bentoml._internal.server;
bentoml._internal.server.start_prod_api_server("{bento_identifier}", port={port}, host="{host}", working_dir="{working_dir}", instance_id=$(CIRCUS.WID), runner_map={cmd_runner_arg}, backlog={backlog});
"""

SCRIPT_NGROK = """
import bentoml._internal.server;
bentoml._internal.server.start_ngrok_server();
"""

SCRIPT_API_SERVER_DEBUG = """
import bentoml._internal.server;
bentoml._internal.server.start_dev_api_server("{bento_identifier}", port={port}, host="{host}", working_dir="{working_dir}", reload={reload}, reload_delay={reload_delay}, instance_id=$(CIRCUS.WID));
"""


def CMD(*args: str) -> str:
    """
    Wrap a command for different OS.
    """
    return " ".join([escape_for_cmd(arg) for arg in args])


def escape_for_cmd(s: str) -> str:
    """
    Escape a string for use in a posix shell/windows batch command.
    """
    lines = s.strip().split("\n")
    lines = [line.strip().replace("\\", "\\\\").replace('"', '\\"') for line in lines]
    cmd = "".join(lines)
    return f'"{cmd}"'


@inject
def _ensure_prometheus_dir(
    prometheus_multiproc_dir: str = Provide[
        DeploymentContainer.prometheus_multiproc_dir
    ],
    clean: bool = True,
):
    if os.path.exists(prometheus_multiproc_dir):
        if not os.path.isdir(prometheus_multiproc_dir):
            shutil.rmtree(prometheus_multiproc_dir)
        elif clean or os.listdir(prometheus_multiproc_dir):
            shutil.rmtree(prometheus_multiproc_dir)
    os.makedirs(prometheus_multiproc_dir, exist_ok=True)


@inject
def serve_development(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[DeploymentContainer.api_server_config.port],
    host: str = Provide[DeploymentContainer.api_server_config.host],
    with_ngrok: bool = False,
    reload: bool = False,
    reload_delay: float = 0.25,
) -> None:
    logger.info('Starting development BentoServer from "%s"', bento_identifier)
    working_dir = os.path.realpath(os.path.expanduser(working_dir))

    from circus.watcher import Watcher  # type: ignore

    env = dict(os.environ)

    watchers: t.List[Watcher] = []

    if with_ngrok:
        watchers.append(
            Watcher(
                name="ngrok",
                cmd=CMD(sys.executable, "-c", SCRIPT_NGROK),
                env=env,
                numprocesses=1,
                stop_children=True,
                working_dir=working_dir,
            )
        )

    cmd_api_server = SCRIPT_API_SERVER_DEBUG.format(
        bento_identifier=bento_identifier,
        port=port,
        host=host,
        working_dir=working_dir,
        reload=reload,
        reload_delay=reload_delay,
    )

    watchers.append(
        Watcher(
            name="api_server",
            cmd=CMD(sys.executable, "-c", cmd_api_server),
            env=env,
            numprocesses=1,
            stop_children=True,
            working_dir=working_dir,
        )
    )

    arbiter = create_standalone_arbiter(watchers)
    _ensure_prometheus_dir()
    arbiter.start()


def start_ngrok_server() -> None:
    from bentoml._internal.utils.flask_ngrok import start_ngrok

    time.sleep(1)
    start_ngrok(DeploymentContainer.api_server_config.port.get())


MAX_AF_UNIX_PATH_LENGTH = 103


@inject
def serve_production(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[DeploymentContainer.api_server_config.port],
    host: str = Provide[DeploymentContainer.api_server_config.host],
    backlog: int = Provide[DeploymentContainer.api_server_config.backlog],
    app_workers: t.Optional[int] = None,
) -> None:
    logger.info('Starting production BentoServer from "%s"', bento_identifier)
    working_dir = os.path.realpath(os.path.expanduser(working_dir))

    svc = load(bento_identifier, working_dir=working_dir)
    env = dict(os.environ)

    import json

    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    runner_bind_map: t.Dict[str, str] = {}

    uds_path = None

    if not psutil.POSIX:
        # use AF_UNIX sockets for Circus
        uds_path = tempfile.mkdtemp()
        for runner_name, runner in svc.runners.items():
            sockets_path = os.path.join(uds_path, f"{id(runner)}.sock")
            assert len(sockets_path) < MAX_AF_UNIX_PATH_LENGTH
            bind = path_to_uri(sockets_path)

            runner_bind_map[runner_name] = bind

            cmd_runner = SCRIPT_RUNNER.format(
                bento_identifier=bento_identifier,
                runner_name=runner_name,
                working_dir=working_dir,
                bind=bind,
            )
            watchers.append(
                Watcher(
                    name=f"runner_{runner_name}",
                    cmd=CMD(sys.executable, "-c", cmd_runner),
                    env=env,
                    numprocesses=runner.num_replica,
                    stop_children=True,
                    working_dir=working_dir,
                )
            )

    elif not psutil.WINDOWS:
        # Windows doesn't (fully) support AF_UNIX sockets
        with contextlib.ExitStack() as port_stack:
            for runner_name, runner in svc.runners.items():
                runner_port = port_stack.enter_context(reserve_free_port())
                bind = f"tcp://127.0.0.1:{runner_port}"

                runner_bind_map[runner_name] = bind

                cmd_runner = SCRIPT_RUNNER.format(
                    bento_identifier=bento_identifier,
                    runner_name=runner_name,
                    working_dir=working_dir,
                    bind=bind,
                )
                watchers.append(
                    Watcher(
                        name=f"runner_{runner_name}",
                        cmd=CMD(sys.executable, "-c", cmd_runner),
                        env=env,
                        numprocesses=runner.num_replica,
                        stop_children=True,
                        working_dir=working_dir,
                    )
                )

            port_stack.enter_context(
                reserve_free_port()
            )  # reserve one more port to avoid conflicts
    else:
        raise NotImplementedError("Unsupported platform: {}".format(sys.platform))

    logger.debug("Runner map: %s", runner_bind_map)

    cmd_api_server = SCRIPT_API_SERVER.format(
        bento_identifier=bento_identifier,
        host=host,
        port=port,
        cmd_runner_arg=json.dumps(runner_bind_map),
        runner_name="api_server",
        working_dir=working_dir,
        backlog=backlog,
    )
    watchers.append(
        Watcher(
            name="api_server",
            cmd=CMD(sys.executable, "-c", cmd_api_server),
            env=env,
            numprocesses=app_workers or 1,
            stop_children=True,
            working_dir=working_dir,
        )
    )

    arbiter = create_standalone_arbiter(watchers=watchers)

    _ensure_prometheus_dir()
    try:
        arbiter.start()
    finally:
        if uds_path is not None:
            shutil.rmtree(uds_path)


def start_dev_api_server(
    bento_identifier: str,
    port: int,
    host: str,
    working_dir: t.Optional[str] = None,
    reload: bool = False,
    reload_delay: t.Optional[float] = None,
    instance_id: t.Optional[int] = None,  # pylint: disable=unused-argument
):
    import uvicorn  # type: ignore

    log_level = "debug" if get_debug_mode() else "info"
    svc = load(bento_identifier, working_dir=working_dir)
    uvicorn_options = {
        "host": host,
        "port": port,
        "log_level": log_level,
        "reload": reload,
        "reload_delay": reload_delay,
        "log_config": UVICORN_LOGGING_CONFIG,
        "workers": 1,
    }

    if reload:
        # When reload=True, the app parameter in uvicorn.run(app) must be the import str
        asgi_app_import_str = f"{svc._import_str}.asgi_app"  # type: ignore[reportPrivateUsage]
        # TODO: use svc.build_args.include/exclude as default files to watch
        # TODO: watch changes in model store when "latest" model tag is used
        uvicorn.run(asgi_app_import_str, **uvicorn_options)
    else:
        uvicorn.run(svc.asgi_app, **uvicorn_options)  # type: ignore


def start_prod_api_server(
    bento_identifier: str,
    port: int,
    host: str,
    runner_map: t.Dict[str, str],
    backlog: int,
    working_dir: t.Optional[str] = None,
    instance_id: t.Optional[int] = None,  # pylint: disable=unused-argument
):
    import uvicorn  # type: ignore

    log_level = "info"
    DeploymentContainer.remote_runner_mapping.set(runner_map)
    svc = load(bento_identifier, working_dir=working_dir)
    uvicorn_options = {
        "host": host,
        "port": port,
        "log_level": log_level,
        "backlog": backlog,
        "log_config": UVICORN_LOGGING_CONFIG,
        "workers": 1,
    }
    uvicorn.run(svc.asgi_app, **uvicorn_options)  # type: ignore


def start_prod_runner_server(
    bento_identifier: str,
    name: str,
    bind: str,
    working_dir: t.Optional[str] = None,
    instance_id: t.Optional[int] = None,
):
    import uvicorn  # type: ignore

    from bentoml._internal.server.runner_app import RunnerAppFactory

    svc = load(bento_identifier, working_dir=working_dir)
    runner = svc.runners[name]
    app = RunnerAppFactory(runner, instance_id=instance_id)()

    parsed = urlparse(bind)
    uvicorn_options = {
        "log_level": "info",
        "log_config": UVICORN_LOGGING_CONFIG,
        "workers": 1,
    }
    if parsed.scheme == "file":
        uvicorn.run(
            app,  # type: ignore
            uds=uri_to_path(bind),
            **uvicorn_options,
        )
    elif parsed.scheme == "tcp":
        uvicorn.run(
            app,  # type: ignore
            host=parsed.hostname,
            port=parsed.port,
            **uvicorn_options,
        )
    else:
        raise ValueError(f"Unsupported bind scheme: {bind}")
