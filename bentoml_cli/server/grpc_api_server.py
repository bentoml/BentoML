from __future__ import annotations

import json
from urllib.parse import urlparse

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option(
    "--bind",
    type=click.STRING,
    required=True,
    help="Bind address sent to circus. This address accepts the following values: 'tcp://127.0.0.1:3000','unix:///tmp/bento_api.sock', 'fd://12'",
)
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
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
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
    help="Enable reflection.",
    default=False,
)
@click.option(
    "--max-concurrent-streams",
    type=click.INT,
    default=None,
    help="Maximum number of concurrent incoming streams to allow on a HTTP/2 connection.",
    show_default=True,
)
def main(
    bento_identifier: str,
    bind: str,
    runner_map: str | None,
    working_dir: str | None,
    worker_id: int | None,
    prometheus_dir: str | None,
    enable_reflection: bool,
    max_concurrent_streams: int,
):
    """
    Start BentoML API server.
    \b
    This is an internal API, users should not use this directly. Instead use `bentoml serve <path> [--options]`
    """

    import bentoml
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.context import component_context
    from bentoml._internal.configuration.containers import BentoMLContainer

    configure_server_logging()

    BentoMLContainer.development_mode.set(False)
    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)

    component_context.component_name = f"grpc_api_server:{worker_id}"

    if runner_map is not None:
        BentoMLContainer.remote_runner_mapping.set(json.loads(runner_map))
    svc = bentoml.load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    if svc.tag is None:
        component_context.bento_name = f"*{svc.__class__.__name__}"
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version

    parsed = urlparse(bind)
    assert parsed.scheme == "tcp"

    from bentoml._internal.server.grpc.config import Config
    from bentoml._internal.server.grpc.server import Server

    grpc_options = {
        "enable_reflection": enable_reflection,
        "max_concurrent_streams": max_concurrent_streams,
        "bind_address": parsed.netloc,
    }

    Server(Config(svc.grpc_servicer, **grpc_options)).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
