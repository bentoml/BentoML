from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import click
import psutil

from bentoml import load
from bentoml._internal.log import configure_server_logging
from bentoml._internal.context import component_context
from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ...grpc_server import GRPCServer


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--bind", type=click.STRING, required=True)
@click.option("--working-dir", required=False, type=click.Path(), default=None)
@click.option(
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
)
def main(
    bento_identifier: str,
    bind: str,
    working_dir: str | None,
    prometheus_dir: t.Optional[str],
):
    import asyncio

    component_context.component_name = "grpc_dev_api_server"

    configure_server_logging()

    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)

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

    cleanup: list[t.Coroutine[t.Any, t.Any, None]] = []

    loop = asyncio.get_event_loop()

    async def start_grpc_server(srv: GRPCServer) -> None:
        srv.add_insecure_port(bind)

        await srv.start()
        cleanup.append(srv.stop())
        await srv.wait_for_termination()

    try:
        loop.run_until_complete(start_grpc_server(svc.grpc_server))
    finally:
        loop.run_until_complete(*cleanup)
        loop.close()


if __name__ == "__main__":
    main()
