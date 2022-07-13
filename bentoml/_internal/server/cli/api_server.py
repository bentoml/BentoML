from __future__ import annotations

import sys
import json
import socket
import typing as t
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import psutil

import bentoml

from ...context import component_context

if TYPE_CHECKING:
    from asgiref.typing import ASGI3Application

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option(
    "--bind",
    type=click.STRING,
    required=True,
    help="Bind address sent to circus. This address accepts the following values: 'tcp://127.0.0.1:3000','unix:///tmp/bento_api.sock', 'fd://12'",
)
@click.option(
    "--runner-map",
    type=click.STRING,
    envvar="BENTOML_RUNNER_MAP",
    help="JSON string of runners map, default sets to envars `BENTOML_RUNNER_MAP`",
)
@click.option(
    "--backlog", type=click.INT, default=2048, help="Backlog size for the socket"
)
@click.option(
    "--working-dir",
    type=click.Path(exists=True),
    help="Working directory for the API server",
)
@click.option(
    "--worker-id",
    required=False,
    type=click.INT,
    default=None,
    help="If set, start the server as a bare worker with the given worker ID. Otherwise start a standalone server with a supervisor process.",
)
@click.pass_context
def main(
    ctx: click.Context,
    bento_identifier: str,
    bind: str,
    runner_map: str | None,
    backlog: int,
    working_dir: str | None,
    worker_id: int | None,
):
    """
    Start BentoML API server.
    \b
    This is an internal API, users should not use this directly. Instead use `bentoml serve <path> [--options]`
    """
    if worker_id is None:
        # Start a standalone server with a supervisor process
        from circus.watcher import Watcher

        from bentoml._internal.server import ensure_prometheus_dir
        from bentoml._internal.utils.click import unparse_click_params
        from bentoml._internal.utils.circus import create_standalone_arbiter
        from bentoml._internal.utils.circus import create_circus_socket_from_uri

        ensure_prometheus_dir()
        circus_socket = create_circus_socket_from_uri(bind, name="_bento_api_server")
        params = ctx.params
        params["bind"] = "fd://$(circus.sockets._bento_api_server)"
        params["worker_id"] = "$(circus.wid)"
        watcher = Watcher(
            name="bento_api_server",
            cmd=sys.executable,
            args=["-m", "bentoml._internal.server.cli.api_server"]
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

    component_context.component_name = f"api_server:{worker_id}"

    from ...log import configure_server_logging
    from ...configuration.containers import BentoMLContainer

    BentoMLContainer.development_mode.set(False)
    configure_server_logging()

    import uvicorn  # type: ignore

    if runner_map is not None:
        BentoMLContainer.remote_runner_mapping.set(json.loads(runner_map))
    svc = bentoml.load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    if svc.tag is None:
        component_context.bento_name = f"*{svc.__class__.__name__}"
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version

    parsed = urlparse(bind)
    uvicorn_options: dict[str, t.Any] = {
        "backlog": backlog,
        "log_config": None,
        "workers": 1,
    }
    if psutil.WINDOWS:
        uvicorn_options["loop"] = "asyncio"
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    app = t.cast("ASGI3Application", svc.asgi_app)
    assert parsed.scheme == "fd"

    # skip the uvicorn internal supervisor
    fd = int(parsed.netloc)
    sock = socket.socket(fileno=fd)
    config = uvicorn.Config(app, **uvicorn_options)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
