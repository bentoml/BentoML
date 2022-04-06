import json
import socket
import typing as t
from typing import Any
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from bentoml import load
from bentoml._internal.server import ensure_prometheus_dir
from bentoml._internal.utils.uri import uri_to_path

from ...log import LOGGING_CONFIG
from ...trace import ServiceContext

if TYPE_CHECKING:
    from asgiref.typing import ASGI3Application

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING)
@click.argument("bind", type=click.STRING)
@click.option("--runner-map", type=click.STRING, envvar="BENTOML_RUNNER_MAP")
@click.option("--backlog", type=click.INT, default=2048)
@click.option("--working-dir", type=click.Path(exists=True))
@click.option(
    "--as-worker",
    required=False,
    type=click.BOOL,
    is_flag=True,
    default=False,
)
def main(
    bento_identifier: str = "",
    bind: str = "",
    runner_map: t.Optional[str] = None,
    backlog: int = 2048,
    working_dir: t.Optional[str] = None,
    as_worker: bool = False,
):
    """
    Start BentoML API server.

    Args
    ----
    bento_identifier: str
        BentoML identifier.
    bind: str
        Bind address.
    runner_map: str
        Path to runner map file.
    backlog: int
        Backlog size.
    working_dir: str
        Working directory.
    as_worker: bool (default: False)
        If True, start the server as a bare worker. Else, start a standalone server
        with a supervisor process.
    """
    if not as_worker:
        # if not as_worker, ensure_prometheus_dir may cause a race.
        ensure_prometheus_dir()

    import uvicorn  # type: ignore

    ServiceContext.component_name_var.set("api_server")

    log_level = "info"
    if runner_map is not None:
        from ...configuration.containers import DeploymentContainer

        DeploymentContainer.remote_runner_mapping.set(json.loads(runner_map))
    svc = load(bento_identifier, working_dir=working_dir, change_global_cwd=True)

    parsed = urlparse(bind)
    uvicorn_options: dict[str, Any] = {
        "log_level": log_level,
        "backlog": backlog,
        "log_config": LOGGING_CONFIG,
        "workers": 1,
    }
    app = t.cast("ASGI3Application", svc.asgi_app)
    if parsed.scheme in ("file", "unix"):
        path = uri_to_path(bind)
        uvicorn_options["uds"] = path
        config = uvicorn.Config(app, **uvicorn_options)
        uvicorn.Server(config).run()
    elif parsed.scheme == "tcp":
        uvicorn_options["host"] = parsed.hostname
        uvicorn_options["port"] = parsed.port
        config = uvicorn.Config(app, **uvicorn_options)
        uvicorn.Server(config).run()
    elif parsed.scheme == "fd":
        # when fd is provided, we will skip the uvicorn internal supervisor, thus there is only one process
        fd = int(parsed.netloc)
        sock = socket.socket(fileno=fd)
        config = uvicorn.Config(app, **uvicorn_options)
        uvicorn.Server(config).run(sockets=[sock])
    else:
        raise ValueError(f"Unsupported bind scheme: {bind}")


if __name__ == "__main__":
    main()
