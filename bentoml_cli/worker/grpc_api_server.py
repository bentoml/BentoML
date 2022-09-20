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
    default=False,
)
@click.option(
    "--max-concurrent-streams",
    type=click.INT,
    help="Maximum number of concurrent incoming streams to allow on a HTTP2 connection.",
    default=None,
)
def main(
    bento_identifier: str,
    host: str,
    port: int,
    runner_map: str | None,
    working_dir: str | None,
    worker_id: int | None,
    enable_reflection: bool,
    max_concurrent_streams: int | None,
):
    """
    Start BentoML API server.
    \b
    This is an internal API, users should not use this directly. Instead use `bentoml serve-grpc <path> [--options]`
    """

    import bentoml
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.context import component_context
    from bentoml._internal.configuration.containers import BentoMLContainer

    component_context.component_type = "grpc_api_server"
    component_context.component_index = worker_id
    configure_server_logging()

    BentoMLContainer.development_mode.set(False)
    if runner_map is not None:
        BentoMLContainer.remote_runner_mapping.set(json.loads(runner_map))

    svc = bentoml.load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    component_context.component_name = svc.name
    if svc.tag is None:
        component_context.bento_name = svc.name
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version or "not available"

    from bentoml._internal.server import grpc

    grpc_options: dict[str, t.Any] = {"enable_reflection": enable_reflection}
    if max_concurrent_streams:
        grpc_options["max_concurrent_streams"] = int(max_concurrent_streams)

    grpc.Server(
        grpc.Config(svc.grpc_servicer, bind_address=f"{host}:{port}", **grpc_options)
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
