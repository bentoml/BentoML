from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from bentoml._internal import external_typing as ext


class GenerateLatestMiddleware:
    def __init__(self, app: ext.ASGIApp):
        from bentoml._internal.configuration.containers import BentoMLContainer

        self.app = app
        self.metrics_client = BentoMLContainer.metrics_client.get()

    async def __call__(
        self, scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
    ) -> None:
        assert scope["type"] == "http"
        assert scope["path"] == "/"

        from starlette.responses import Response

        return await Response(
            self.metrics_client.generate_latest(),
            status_code=200,
            media_type=self.metrics_client.CONTENT_TYPE_LATEST,
        )(scope, receive, send)


@click.command()
@click.option("--fd", type=click.INT, required=True)
@click.option("--backlog", type=click.INT, default=2048)
@click.option(
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
)
def main(fd: int, backlog: int, prometheus_dir: str | None):
    """
    Start a standalone Prometheus server to use with gRPC.
    \b
    This is an internal API, users should not use this directly. Instead use 'bentoml serve-grpc'.
    Prometheus then can be accessed at localhost:9090
    """

    import socket

    import psutil
    import uvicorn
    from starlette.middleware import Middleware
    from starlette.applications import Starlette
    from starlette.middleware.wsgi import WSGIMiddleware  # TODO: a2wsgi

    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.context import component_context
    from bentoml._internal.configuration import get_debug_mode
    from bentoml._internal.configuration.containers import BentoMLContainer

    component_context.component_type = "prom_server"

    configure_server_logging()

    BentoMLContainer.development_mode.set(False)
    metrics_client = BentoMLContainer.metrics_client.get()
    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)

    # create a ASGI app that wraps around the default HTTP prometheus server.
    prom_app = Starlette(
        debug=get_debug_mode(), middleware=[Middleware(GenerateLatestMiddleware)]
    )
    prom_app.mount("/", WSGIMiddleware(metrics_client.make_wsgi_app()))
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
    uvicorn.Server(uvicorn.Config(prom_app, **uvicorn_options)).run(sockets=[sock])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
