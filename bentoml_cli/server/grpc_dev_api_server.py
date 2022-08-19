from __future__ import annotations

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--bind", type=click.STRING, required=True)
@click.option("--working-dir", required=False, type=click.Path(), default=None)
@click.option(
    "--enable-reflection",
    type=click.BOOL,
    help="Enable reflection.",
    default=False,
)
def main(
    bento_identifier: str,
    bind: str,
    working_dir: str | None,
    enable_reflection: bool,
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

        grpc_options = {"enable_reflection": enable_reflection}

        config = grpc.Config(bind_address=f"[::]:{parsed.port}", **grpc_options)
        print(config.options)
        grpc.Server(config, svc.grpc_servicer).run()
    else:
        raise ValueError(f"Unsupported bind scheme: {bind}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
