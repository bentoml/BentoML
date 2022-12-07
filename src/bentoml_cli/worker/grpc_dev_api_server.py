from __future__ import annotations

import typing as t

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--host", type=click.STRING, required=False, default=None)
@click.option("--port", type=click.INT, required=False, default=None)
@click.option("--working-dir", required=False, type=click.Path(), default=None)
@click.option(
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
)
@click.option(
    "--enable-reflection",
    type=click.BOOL,
    is_flag=True,
    help="Enable reflection.",
)
@click.option(
    "--enable-channelz",
    type=click.BOOL,
    is_flag=True,
    help="Enable channelz.",
    default=False,
)
@click.option(
    "--max-concurrent-streams",
    type=int,
    help="Maximum number of concurrent incoming streams to allow on a http2 connection.",
    default=None,
)
@click.option(
    "--ssl-certfile",
    type=str,
    default=None,
    help="SSL certificate file",
)
@click.option(
    "--ssl-keyfile",
    type=str,
    default=None,
    help="SSL key file",
)
@click.option(
    "--ssl-ca-certs",
    type=str,
    default=None,
    help="CA certificates file",
)
@click.option(
    "--protocol-version",
    type=click.Choice(["v1", "v1alpha1"]),
    help="Determine the version of generated gRPC stubs to use.",
    default="v1",
    show_default=True,
)
def main(
    bento_identifier: str,
    host: str,
    port: int,
    prometheus_dir: str | None,
    working_dir: str | None,
    enable_reflection: bool,
    enable_channelz: bool,
    max_concurrent_streams: int | None,
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
    ssl_ca_certs: str | None,
    protocol_version: str,
):
    import psutil

    from bentoml import load
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.context import component_context
    from bentoml._internal.configuration.containers import BentoMLContainer

    component_context.component_type = "grpc_dev_api_server"
    configure_server_logging()
    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)

    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)
    if not port:
        port = BentoMLContainer.grpc.port.get()
    if not host:
        host = BentoMLContainer.grpc.host.get()

    # setup context
    component_context.component_name = svc.name
    if svc.tag is None:
        component_context.bento_name = svc.name
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version or "not available"
    if psutil.WINDOWS:
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    from bentoml._internal.server import grpc_app as grpc

    grpc_options: dict[str, t.Any] = {
        "bind_address": f"{host}:{port}",
        "enable_reflection": enable_reflection,
        "enable_channelz": enable_channelz,
        "protocol_version": protocol_version,
    }
    if max_concurrent_streams is not None:
        grpc_options["max_concurrent_streams"] = int(max_concurrent_streams)
    if ssl_certfile:
        grpc_options["ssl_certfile"] = ssl_certfile
    if ssl_keyfile:
        grpc_options["ssl_keyfile"] = ssl_keyfile
    if ssl_ca_certs:
        grpc_options["ssl_ca_certs"] = ssl_ca_certs

    grpc.Server(svc, **grpc_options).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
