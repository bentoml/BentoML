import typing as t
import asyncio
from urllib.parse import urlparse

import click

from bentoml import load
from bentoml.exceptions import MissingDependencyException

from ...context import component_context

try:
    import grpc
except ImportError:
    raise MissingDependencyException(
        "`grpcio` is required for `--grpc`. Install requirements with `pip install 'bentoml[grpc]'`."
    )

from bentoml.protos import service_pb2_grpc


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--bind", type=click.STRING, required=True)
@click.option("--working-dir", required=False, type=click.Path(), default=None)
@click.option("--port", type=click.INT, required=False)
def main(
    bento_identifier: str,
    bind: str,
    working_dir: t.Optional[str],
    port: int,
):
    component_context.component_name = "grpc_dev_api_server"

    from ...log import configure_server_logging

    configure_server_logging()

    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    if svc.tag is None:
        component_context.bento_name = f"*{svc.__class__.__name__}"
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version

    parsed = urlparse(bind)

    if parsed.scheme == "fd":
        # initialize runners
        for runner in svc.runners:
            runner.init_local()

        _cleanup_coroutines: list[t.Coroutine[t.Any, t.Any, None]] = []

        async def serve() -> None:
            server = grpc.aio.server()
            grpc_servicer = svc.get_grpc_servicer()
            service_pb2_grpc.add_BentoServiceServicer_to_server(grpc_servicer(), server)
            listen_addr = "[::]:" + str(port)
            server.add_insecure_port(listen_addr)

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
    else:
        raise ValueError(f"Unsupported bind scheme: {bind}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
