import sys
import json
import socket
import typing as t
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import bentoml

from ...log import LOGGING_CONFIG
from ...trace import ServiceContext

if TYPE_CHECKING:
    from asgiref.typing import ASGI3Application

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--bind", type=click.STRING, required=True)
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
@click.pass_context
def main(
    ctx: click.Context,
    bento_identifier: str,
    bind: str,
    runner_map: t.Optional[str],
    backlog: int,
    working_dir: t.Optional[str],
    as_worker: bool,
):
    """
    Start BentoML API server.

    Args
    ----
    bento_identifier: str
        BentoML identifier.
    bind: str
        Bind address.
        values:
            - "tcp://127.0.0.1:3000"
            - "unix:///tmp/bento_api.sock"
            - "fd://12"
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
        params["as_worker"] = True
        watcher = Watcher(
            name="bento_api_server",
            cmd=sys.executable,
            args=["-m", "bentoml._internal.server.cli.api_server"]
            + unparse_click_params(params, ctx.command.params),
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

    ServiceContext.component_name_var.set("api_server")

    log_level = "info"
    if runner_map is not None:
        from ...configuration.containers import DeploymentContainer

        DeploymentContainer.remote_runner_mapping.set(json.loads(runner_map))
    svc = bentoml.load(
        bento_identifier, working_dir=working_dir, change_global_cwd=True
    )

    parsed = urlparse(bind)
    uvicorn_options: dict[str, t.Any] = {
        "log_level": log_level,
        "backlog": backlog,
        "log_config": LOGGING_CONFIG,
        "workers": 1,
    }
    app = t.cast("ASGI3Application", svc.asgi_app)
    assert parsed.scheme == "fd"

    # skip the uvicorn internal supervisor
    fd = int(parsed.netloc)
    sock = socket.socket(fileno=fd)
    config = uvicorn.Config(app, **uvicorn_options)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
