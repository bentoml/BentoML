import sys
import socket
import typing as t
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import psutil

from bentoml import load
from bentoml._internal.utils.uri import uri_to_path

from ...log import LOGGING_CONFIG
from ...trace import ServiceContext

if TYPE_CHECKING:
    from asgiref.typing import ASGI3Application

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--runner-name", type=click.STRING, required=True)
@click.option("--bind", type=click.STRING, required=True)
@click.option("--working-dir", required=False, default=None, help="Working directory")
@click.option(
    "--as-worker",
    required=False,
    type=click.BOOL,
    is_flag=True,
    default=False,
)
@click.option(
    "--worker-index",
    type=click.INT,
    required=False,
    default=0,
    help="Worker index",
)
@click.pass_context
def main(
    ctx: click.Context,
    bento_identifier: str,
    runner_name: str,
    bind: str,
    working_dir: t.Optional[str],
    as_worker: bool,
    worker_index: int,
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
        as_worker: (Optional) if True, the runner will be started as a worker
    """

    if not as_worker:
        # Start a standalone server with a supervisor process
        from circus.watcher import Watcher

        from bentoml._internal.utils.click import unparse_click_params
        from bentoml._internal.utils.circus import create_standalone_arbiter
        from bentoml._internal.utils.circus import create_circus_socket_from_uri

        circus_socket = create_circus_socket_from_uri(bind, name=runner_name)
        params = ctx.params
        params["bind"] = f"fd://$(circus.sockets.{runner_name})"
        params["as_worker"] = True
        watcher = Watcher(
            name=f"runner_{runner_name}",
            cmd=sys.executable,
            args=["-m", "bentoml._internal.server.cli.runner"]
            + unparse_click_params(params, ctx.command.params, factory=str),
            copy_env=True,
            numprocesses=1,
            stop_children=True,
            use_sockets=True,
            working_dir=working_dir,
        )
        arbiter = create_standalone_arbiter(watchers=[watcher], sockets=[circus_socket])
        arbiter.start()
        return

    import uvicorn  # type: ignore

    from bentoml._internal.server.runner_app import RunnerAppFactory

    ServiceContext.component_name_var.set(runner_name)

    service = load(bento_identifier, working_dir=working_dir, change_global_cwd=True)
    for runner in service.runners:
        if runner.name == runner_name:
            break
    else:
        raise ValueError(f"Runner {runner_name} not found")

    app = t.cast(
        "ASGI3Application", RunnerAppFactory(runner, worker_index=worker_index)()
    )

    parsed = urlparse(bind)
    uvicorn_options = {
        "log_level": "info",
        "log_config": LOGGING_CONFIG,
        "workers": 1,
    }

    if psutil.WINDOWS:
        uvicorn_options["loop"] = "asyncio"
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

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
    main()  # pylint: disable=no-value-for-parameter
