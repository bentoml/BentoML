import socket
import typing as t
from urllib.parse import urlparse

import click
import psutil

from bentoml import load

from ...log import LOGGING_CONFIG
from ...trace import ServiceContext


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--bind", type=click.STRING, required=True)
@click.option("--backlog", type=click.INT, default=2048)
@click.option("--working-dir", required=False, type=click.Path(), default=None)
def main(
    bento_identifier: str,
    bind: str,
    working_dir: t.Optional[str],
    backlog: int,
):
    import uvicorn  # type: ignore

    from ...configuration import get_debug_mode

    ServiceContext.component_name_var.set("dev_api_server")
    parsed = urlparse(bind)

    if parsed.scheme == "fd":
        fd = int(parsed.netloc)
        sock = socket.socket(fileno=fd)
        log_level = "debug" if get_debug_mode() else "info"
        svc = load(bento_identifier, working_dir=working_dir, change_global_cwd=True)
        uvicorn_options = {
            "log_level": log_level,
            "backlog": backlog,
            "log_config": LOGGING_CONFIG,
            "workers": 1,
            "lifespan": "on",
        }
        if psutil.WINDOWS:
            uvicorn_options["loop"] = "asyncio"
            import asyncio

            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

        config = uvicorn.Config(svc.asgi_app, **uvicorn_options)
        uvicorn.Server(config).run(sockets=[sock])
    else:
        raise ValueError(f"Unsupported bind scheme: {bind}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
