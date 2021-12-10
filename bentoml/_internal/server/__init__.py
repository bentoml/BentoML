import os
import sys
import time
import shutil
import typing as t
import logging
import tempfile

from simple_di import inject
from simple_di import Provide

from bentoml import load

from ..configuration import get_debug_mode
from ..configuration.containers import BentoServerContainer

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


@inject
def serve_development(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoServerContainer.config.port],
    host: str = Provide[BentoServerContainer.config.host],
    with_ngrok: bool = False,
    reload: bool = False,
    reload_delay: float = 0.25,
) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_dir))

    from circus.util import DEFAULT_ENDPOINT_SUB  # type: ignore
    from circus.util import DEFAULT_ENDPOINT_DEALER  # type: ignore
    from circus.arbiter import Arbiter  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    env = dict(os.environ)

    watchers: t.List[Watcher] = []
    if with_ngrok:
        cmd_ngrok = """
import bentoml._internal.server
bentoml._internal.server.start_ngrok_server()"""
        watchers.append(
            Watcher(
                name="ngrok",
                cmd=f"{sys.executable} -c '{cmd_ngrok}'",
                env=env,
                numprocesses=1,
                stop_children=True,
                working_dir=working_dir,
            )
        )

    cmd_api_server = f"""
import bentoml._internal.server
bentoml._internal.server.start_dev_api_server(
    "{bento_identifier}",
    port={port},
    host="{host}",
    working_dir="{working_dir}",
    reload={reload},
    reload_delay={reload_delay},
    instance_id=$(CIRCUS.WID),
)"""
    watchers.append(
        Watcher(
            name="api_server",
            cmd=f"{sys.executable} -c '{cmd_api_server}'",
            env=env,
            numprocesses=1,
            stop_children=True,
            working_dir=working_dir,
        )
    )

    arbiter = Arbiter(
        watchers=watchers,
        endpoint=DEFAULT_ENDPOINT_DEALER,
        pubsub_endpoint=DEFAULT_ENDPOINT_SUB,
    )

    arbiter.start()


def start_ngrok_server() -> None:
    from bentoml._internal.utils.flask_ngrok import start_ngrok

    time.sleep(1)
    start_ngrok(BentoServerContainer.config.port.get())


MAX_AF_UNIX_PATH_LENGTH = 103


@inject
def serve_production(
    bento_identifier: str,
    working_dir: str,
    port: int = Provide[BentoServerContainer.config.port],
    host: str = Provide[BentoServerContainer.config.host],
    backlog: int = Provide[BentoServerContainer.config.backlog],
    app_workers: t.Optional[int] = None,
) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_dir))

    svc = load(bento_identifier, working_dir=working_dir)
    env = dict(os.environ)

    import json

    from circus.util import DEFAULT_ENDPOINT_SUB  # type: ignore
    from circus.util import DEFAULT_ENDPOINT_DEALER  # type: ignore
    from circus.arbiter import Arbiter  # type: ignore
    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    uds_path = tempfile.mkdtemp()
    watchers: t.List[Watcher] = []
    sockets_map: t.Dict[str, CircusSocket] = {}

    for runner_name, runner in svc.runners.items():
        sockets_path = os.path.join(uds_path, f"{id(runner)}.sock")
        assert len(sockets_path) < MAX_AF_UNIX_PATH_LENGTH
        sockets_map[runner_name] = CircusSocket(
            name=runner_name,
            path=sockets_path,
            umask=0,
        )
        cmd_runner = f"""import bentoml._internal.server
bentoml._internal.server.start_prod_runner_server(
    "{bento_identifier}",
    "{runner_name}",
    working_dir="{working_dir}",
    instance_id=$(CIRCUS.WID),
    fd=$(circus.sockets.{runner_name}),
)"""
        watchers.append(
            Watcher(
                name=f"runner_{runner_name}",
                cmd=f"{sys.executable} -c '{cmd_runner}'",
                env=env,
                numprocesses=runner.num_replica,
                stop_children=True,
                use_sockets=True,
                working_dir=working_dir,
            )
        )
    logger.debug("Runner sockets_map: %s", sockets_map)

    cmd_runner_arg = json.dumps({k: f"{v.path}" for k, v in sockets_map.items()})
    cmd_api_server = f"""
import bentoml._internal.server
bentoml._internal.server.start_prod_api_server(
    "{bento_identifier}",
    port={port},
    host="{host}",
    working_dir="{working_dir}",
    instance_id=$(CIRCUS.WID),
    runner_map={cmd_runner_arg},
    backlog={backlog},
)"""
    watchers.append(
        Watcher(
            name="api_server",
            cmd=f"{sys.executable} -c '{cmd_api_server}'",
            env=env,
            numprocesses=app_workers or 1,
            stop_children=True,
            use_sockets=True,
            working_dir=working_dir,
        )
    )

    arbiter = Arbiter(
        watchers=watchers,
        endpoint=DEFAULT_ENDPOINT_DEALER,
        pubsub_endpoint=DEFAULT_ENDPOINT_SUB,
        sockets=[s for s in sockets_map.values()],
    )

    try:
        arbiter.start()
    finally:

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
    BentoServerContainer.remote_runner_mapping.set(runner_map)
    svc = load(bento_identifier, working_dir=working_dir)
    uvicorn_options = {
        "host": host,
        "port": port,
        "log_level": log_level,
        "backlog": backlog,
        "log_config": UVICORN_LOGGING_CONFIG,
    }

    uvicorn.run(svc.asgi_app, **uvicorn_options)  # type: ignore


def start_prod_runner_server(
    bento_identifier: str,
    name: str,
    fd: int,
    working_dir: t.Optional[str] = None,
    instance_id: t.Optional[int] = None,
):
    import uvicorn  # type: ignore

    from bentoml._internal.server.runner_app import RunnerAppFactory

    svc = load(bento_identifier, working_dir=working_dir)
    runner = svc.runners[name]
    app = RunnerAppFactory(runner, instance_id=instance_id)()

    uvicorn.run(
        app, fd=fd, log_level="info", log_config=UVICORN_LOGGING_CONFIG  # type: ignore
    )
