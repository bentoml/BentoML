from __future__ import annotations

import json
import typing as t

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--host", type=click.STRING, required=False, default=None)
@click.option("--port", type=click.INT, required=False, default=None)
@click.option(
    "--runner-map",
    type=click.STRING,
    envvar="BENTOML_RUNNER_MAP",
    help="JSON string of runners map, default sets to envars `BENTOML_RUNNER_MAP`",
)
@click.option(
    "--working-dir",
    type=click.Path(exists=True),
    help="Working directory for the API server",
)
@click.option("--args", type=click.STRING, help="Bento arguments dict for the service")
@click.option(
    "--worker-id",
    required=False,
    type=click.INT,
    default=None,
    help="If set, start the server as a bare worker with the given worker ID. Otherwise start a standalone server with a supervisor process.",
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
    type=click.INT,
    help="Maximum number of concurrent incoming streams to allow on a HTTP2 connection.",
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
    type=click.Choice(["v1"]),
    help="Determine the version of generated gRPC stubs to use.",
    default="v1",
    show_default=True,
)
@click.option(
    "--development-mode",
    type=click.BOOL,
    help="Run the API server in development mode",
    is_flag=True,
    default=False,
    show_default=True,
)
def main(
    bento_identifier: str,
    host: str,
    port: int,
    runner_map: str | None,
    working_dir: str | None,
    args: str | None,
    worker_id: int | None,
    enable_reflection: bool,
    enable_channelz: bool,
    max_concurrent_streams: int | None,
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
    ssl_ca_certs: str | None,
    protocol_version: str,
    development_mode: bool,
):
    """
    Start a gRPC BentoML API server for a new-style Service.
    \b
    This is an internal API, users should not use this directly. Instead use `bentoml serve-grpc <path> [--options]`
    """

    from _bentoml_impl.loader import load
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.context import server_context
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.utils.args import set_arguments

    server_context.service_type = "grpc_api_server"
    server_context.worker_index = worker_id
    configure_server_logging()

    if worker_id is None:
        BentoMLContainer.config.runner_probe.enabled.set(False)

    BentoMLContainer.development_mode.set(development_mode)
    if runner_map is not None:
        BentoMLContainer.remote_runner_mapping.set(json.loads(runner_map))
    if args:
        set_arguments(json.loads(args))

    svc = load(bento_identifier, working_dir=working_dir)
    svc.inject_config()
    if not port:
        port = BentoMLContainer.grpc.port.get()
    if not host:
        host = BentoMLContainer.grpc.host.get()

    server_context.service_name = svc.name
    if svc.bento is None or svc.bento.tag is None:
        server_context.bento_name = svc.name
        server_context.bento_version = "not available"
    else:
        server_context.bento_name = svc.bento.tag.name
        server_context.bento_version = svc.bento.tag.version or "not available"

    from _bentoml_impl.server import grpc_app as grpc

    grpc_options: dict[str, t.Any] = {
        "bind_address": f"{host}:{port}",
        "enable_reflection": enable_reflection,
        "enable_channelz": enable_channelz,
        "protocol_version": protocol_version,
    }
    if max_concurrent_streams:
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
