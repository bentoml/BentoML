from __future__ import annotations

import typing as t

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--bind", type=click.STRING, required=True)
@click.option("--working-dir", required=False, type=click.Path(), default=None)
@click.option(
    "--enable-reflection",
    type=click.BOOL,
    is_flag=True,
    help="Enable reflection.",
    default=False,
)
@click.option(
    "--max-concurrent-streams",
    type=int,
    help="Maximum number of concurrent incoming streams to allow on a http2 connection.",
    default=None,
)
def main(
    bento_identifier: str,
    bind: str,
    working_dir: str | None,
    enable_reflection: bool,
    max_concurrent_streams: int | None,
):
    from urllib.parse import urlparse

    import psutil

    from bentoml import load
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.context import component_context

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
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    parsed = urlparse(bind)

    if parsed.scheme == "tcp":
        from bentoml._internal.server import grpc

        grpc_options: dict[str, t.Any] = {"enable_reflection": enable_reflection}
        if max_concurrent_streams:
            grpc_options["max_concurrent_streams"] = int(max_concurrent_streams)

        config = grpc.Config(
            svc.grpc_servicer,
            bind_address=f"[::]:{parsed.port}",
            **grpc_options,
        )
        grpc.Server(config).run()
    else:
        raise ValueError(f"Unsupported bind scheme: {bind}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
