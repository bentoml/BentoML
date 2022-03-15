import socket
import typing as t
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from bentoml import load
from bentoml._internal.utils.uri import uri_to_path

from ...log import LOGGING_CONFIG
from ...trace import ServiceContext

if TYPE_CHECKING:
    from asgiref.typing import ASGI3Application

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING)
@click.argument("runner_name", type=click.STRING)
@click.argument("bind", type=click.STRING)
@click.option("--working-dir", required=False, default=None, help="Working directory")
def main(
    bento_identifier: str = "",
    runner_name: str = "",
    bind: str = "",
    working_dir: t.Optional[str] = None,
) -> None:
    """
    Start a runner server.

    Args:
        bento_identifier: the Bento identifier
        name: the name of the runner
        bind: the bind address URI. Can be:
            - tcp://host:port
            - unix://path/to/unix.sock
            - file:///path/to/unix.sock
            - fd://12
        working_dir: (Optional) the working directory
    """

    import uvicorn  # type: ignore

    from bentoml._internal.server.runner_app import RunnerAppFactory

    ServiceContext.component_name_var.set(runner_name)

    svc = load(bento_identifier, working_dir=working_dir, change_global_cwd=True)
    runner = svc.runners[runner_name]
    app = t.cast("ASGI3Application", RunnerAppFactory(runner)())

    parsed = urlparse(bind)
    uvicorn_options = {
        "log_level": "info",
        "log_config": LOGGING_CONFIG,
        "workers": 1,
    }
    if parsed.scheme in ("file", "unix"):
        uvicorn.run(
            app,
            uds=uri_to_path(bind),
            **uvicorn_options,
        )
    elif parsed.scheme == "tcp":
        uvicorn.run(
            app,
            host=parsed.hostname,
            port=parsed.port,
            **uvicorn_options,
        )
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
