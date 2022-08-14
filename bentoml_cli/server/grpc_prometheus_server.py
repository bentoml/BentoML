from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from bentoml._internal import external_typing as ext

logger = logging.getLogger("bentoml")


@click.command()
@click.option("--bind", type=click.STRING, required=True)
@click.option("--backlog", type=click.INT, default=2048)
@click.option(
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
)
def main(bind: str, backlog: int, prometheus_dir: str | None):
    """
    Start BentoML API server.
    \b
    This is an internal API, users should not use this directly. Instead use `bentoml serve <path> [--options]`
    """

    import socket
    from urllib.parse import urlparse

    import psutil
    import uvicorn
    from starlette.middleware import Middleware
    from starlette.applications import Starlette
    from starlette.middleware.wsgi import WSGIMiddleware  # TODO: a2wsgi

    from bentoml.serve import ensure_prometheus_dir
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.context import component_context
    from bentoml._internal.configuration import get_debug_mode
    from bentoml._internal.configuration.containers import BentoMLContainer

    metrics_client = BentoMLContainer.metrics_client.get()

    configure_server_logging()

    BentoMLContainer.development_mode.set(False)

    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)

    ensure_prometheus_dir()

    component_context.component_name = "prom_server"

    class GenerateLatestMiddleware:
        def __init__(self, app: ext.ASGIApp):
            self.app = app

        async def __call__(
            self, scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
        ) -> None:
            assert scope["type"] == "http"
            assert scope["path"] == "/"

            from starlette.responses import Response

            response = Response(
                metrics_client.generate_latest(),
                status_code=200,
                media_type=metrics_client.CONTENT_TYPE_LATEST,
            )

            await response(scope, receive, send)
            return

    # create a ASGI app that wraps around the default HTTP prometheus server.
    middlewares = [Middleware(GenerateLatestMiddleware)]
    prom_app = Starlette(debug=get_debug_mode(), middleware=middlewares)
    prom_app.mount("/", WSGIMiddleware(metrics_client.make_wsgi_app()))

    parsed = urlparse(bind)

    if parsed.scheme == "fd":
        fd = int(parsed.netloc)
        sock = socket.socket(fileno=fd)

        uvicorn_options: dict[str, t.Any] = {
            "backlog": backlog,
            "log_config": None,
            "workers": 1,
        }

        if psutil.WINDOWS:
            uvicorn_options["loop"] = "asyncio"
            import asyncio

            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

        config = uvicorn.Config(prom_app, **uvicorn_options)
        uvicorn.Server(config).run(sockets=[sock])
    else:
        raise ValueError(f"Unsupported bind scheme: {parsed.scheme}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
