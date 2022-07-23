from __future__ import annotations

import typing as t

import click
import psutil
import grpc

from bentoml import load

from bentoml._internal.log import configure_server_logging
from bentoml._internal.context import component_context


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--bind", type=click.STRING, required=True)
@click.option("--working-dir", required=False, type=click.Path(), default=None)
def main(
    bento_identifier: str,
    bind: str,
    working_dir: str | None,
):
    import asyncio

    component_context.component_name = "grpc_dev_api_server"

    configure_server_logging()

    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    if svc.tag is None:
        component_context.bento_name = f"*{svc.__class__.__name__}"
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version

    if psutil.WINDOWS:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    # initialize runners
    # for runner in svc.runners:
    #     runner.init_local()

    _cleanup_coroutines: list[t.Coroutine[t.Any, t.Any, None]] = []

    async def serve() -> None:
        server = grpc.aio.server()
        service_pb2_grpc.add_BentoServiceServicer_to_server(svc.grpc_servicer, server)
        server.add_insecure_port(bind)

        await server.start()

        async def server_graceful_shutdown() -> None:
            # Shuts down the server with 5 seconds of grace period. During the
            # grace period, the server won't accept new connections and allow
            # existing RPCs to continue within the grace period.
            await server.stop(5)

        _cleanup_coroutines.append(server_graceful_shutdown())
        await server.wait_for_termination()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve())
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
