import os
import sys
import shutil
import socket
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
bentoml._internal.server.start_runner_server();
"""

SCRIPT_API_SERVER = """
import bentoml._internal.server;
bentoml._internal.server.start_api_server("{bento_identifier}", port={port}, host="{host}", working_dir="{working_dir}", runner_map={cmd_runner_arg}, backlog={backlog});
"""

SCRIPT_NGROK = """
import bentoml._internal.server;
bentoml._internal.server.start_ngrok_server();
"""

SCRIPT_API_SERVER_DEBUG = """
import bentoml._internal.server;
bentoml._internal.server.start_dev_api_server("{bento_identifier}", port={port}, host="{host}", working_dir="{working_dir}", reload={reload}, reload_delay={reload_delay});
"""


def CMD(*args: str) -> str:
    """
    Wrap a command for different OS.
    """
    return args[0] + " " + " ".join([escape_for_cmd(arg) for arg in args[1:]])


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

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}
    runner_bind_map: t.Dict[str, str] = {}

    uds_path = None

    if psutil.POSIX:
        # use AF_UNIX sockets for Circus
        uds_path = tempfile.mkdtemp()
        for runner_name in svc.runners:
            sockets_path = os.path.join(uds_path, f"{id(runner_name)}.sock")
            assert len(sockets_path) < MAX_AF_UNIX_PATH_LENGTH

            runner_bind_map[runner_name] = path_to_uri(sockets_path)
            circus_socket_map[runner_name] = CircusSocket(
                name=runner_name,
                path=sockets_path,
                backlog=backlog,
            )

            watchers.append(
                Watcher(
                    name=f"runner_{runner_name}",
                    cmd=CMD(
                        sys.executable,
                        "-c",
                        SCRIPT_RUNNER,
                        bento_identifier,
                        runner_name,
                        f"fd://$(circus.sockets.{runner_name})",
                        working_dir,
                    ),
                    env=env,
                    stop_children=True,
                    working_dir=working_dir,
                    use_sockets=True,
                )
            )

    elif psutil.WINDOWS:
        # Windows doesn't (fully) support AF_UNIX sockets
        with contextlib.ExitStack() as port_stack:
            for runner_name in svc.runners:
                runner_port = port_stack.enter_context(reserve_free_port())
                runner_host = "127.0.0.1"

                runner_bind_map[runner_name] = f"tcp://{runner_host}:{runner_port}"
                circus_socket_map[runner_name] = CircusSocket(
                    name=runner_name,
                    host=runner_host,
                    port=runner_port,
                    backlog=backlog,
                )

                watchers.append(
                    Watcher(
                        name=f"runner_{runner_name}",
                        cmd=CMD(
                            sys.executable,
                            "-c",
                            SCRIPT_RUNNER,
                            bento_identifier,
                            runner_name,
                            f"fd://$(circus.sockets.{runner_name})",
                            working_dir,
                        ),
                        env=env,
                        stop_children=True,
                        use_sockets=True,
                        working_dir=working_dir,
                    )
                )
            port_stack.enter_context(
                reserve_free_port()
            )  # reserve one more to avoid conflicts
    else:
        raise NotImplementedError("Unsupported platform: {}".format(sys.platform))

    logger.debug("Runner map: %s", runner_bind_map)

    cmd_api_server = SCRIPT_API_SERVER.format(
        bento_identifier=bento_identifier,
        host=host,
        port=port,
        cmd_runner_arg=json.dumps(runner_bind_map),
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
            use_sockets=True,
            working_dir=working_dir,
        )
    )

    arbiter = create_standalone_arbiter(
        watchers=watchers,
        sockets=list(circus_socket_map.values()),
    )

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


def start_api_server(
    bento_identifier: str,
    port: int,
    host: str,
    runner_map: t.Dict[str, str],
    backlog: int,
    working_dir: t.Optional[str] = None,
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


def start_runner_server():
    """
    Start a runner server.

    Sys Args:
        bento_identifier: the Bento identifier
        name: the name of the runner
        bind: the bind address URI. Can be:
            - tcp://host:port
            - unix://path/to/unix.sock
            - file:///path/to/unix.sock
            - fd://12
            * Note: the fd:// scheme is only supported on Unix systems.
        working_dir: the working directory
    """
    assert len(sys.argv) in (4, 5), "Invalid number of arguments"
    bento_identifier: str = sys.argv[1]
    name: str = sys.argv[2]
    bind: str = sys.argv[3]
    if len(sys.argv) == 5:
        working_dir = sys.argv[4]
    else:
        working_dir = None

    import uvicorn  # type: ignore

    from bentoml._internal.server.runner_app import RunnerAppFactory

    svc = load(bento_identifier, working_dir=working_dir)
    runner = svc.runners[name]
    app = RunnerAppFactory(runner)()

    parsed = urlparse(bind)
    uvicorn_options = {
        "log_level": "info",
        "log_config": UVICORN_LOGGING_CONFIG,
    }
    if parsed.scheme in ("file", "unix"):
        uvicorn.run(
            app,  # type: ignore
            uds=uri_to_path(bind),
            workers=runner.num_replica,
            **uvicorn_options,
        )
    elif parsed.scheme == "tcp":
        uvicorn.run(
            app,  # type: ignore
            host=parsed.hostname,
            port=parsed.port,
            **uvicorn_options,
        )
    elif parsed.scheme == "fd":
        # when fd is provided, we will skip the uvicorn internal supervisor
        fd = int(parsed.netloc)
        sock = socket.socket(fileno=fd)
        config = uvicorn.Config(app, **uvicorn_options)
        uvicorn.Server(config).run(sockets=[sock])
    else:
        raise ValueError(f"Unsupported bind scheme: {bind}")
