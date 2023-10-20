from __future__ import annotations

import json
import typing as t

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option(
    "--fd",
    type=click.INT,
    required=True,
    help="File descriptor of the socket to listen on",
)
@click.option(
    "--runner-map",
    type=click.STRING,
    envvar="BENTOML_RUNNER_MAP",
    help="JSON string of runners map, default sets to envars `BENTOML_RUNNER_MAP`",
)
@click.option(
    "--backlog", type=click.INT, default=2048, help="Backlog size for the socket"
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
    "--ssl-keyfile-password",
    type=str,
    default=None,
    help="SSL keyfile password",
)
@click.option(
    "--ssl-version",
    type=int,
    default=None,
    help="SSL version to use (see stdlib 'ssl' module)",
)
@click.option(
    "--ssl-cert-reqs",
    type=int,
    default=None,
    help="Whether client certificate is required (see stdlib 'ssl' module)",
)
@click.option(
    "--ssl-ca-certs",
    type=str,
    default=None,
    help="CA certificates file",
)
@click.option(
    "--ssl-ciphers",
    type=str,
    default=None,
    help="Ciphers to use (see stdlib 'ssl' module)",
)
@click.option(
    "--development-mode",
    type=click.BOOL,
    help="Run the API server in development mode",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--timeout",
    type=click.INT,
    help="Specify the timeout for API server",
)
def main(
    bento_identifier: str,
    fd: int,
    runner_map: str | None,
    backlog: int,
    working_dir: str | None,
    worker_id: int | None,
    prometheus_dir: str | None,
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
    ssl_keyfile_password: str | None,
    ssl_version: int | None,
    ssl_cert_reqs: int | None,
    ssl_ca_certs: str | None,
    ssl_ciphers: str | None,
    development_mode: bool,
    timeout: int,
):
    """
    Start a HTTP server worker for given service.
    """
    from bentoml._internal.container import BentoMLContainer
    from bentoml._internal.service import load
    from bentoml_io.server import Service
    from bentoml_io.server.serving import serve_http

    service = t.cast(Service, load(bento_identifier, working_dir=working_dir))
    if runner_map:
        BentoMLContainer.remote_runner_mapping.set(
            t.cast(t.Dict[str, str], json.loads(runner_map))
        )

    serve_http(
        service,
        fd=fd,
        timeout=timeout,
        worker_id=worker_id,
        backlog=backlog,
        prometheus_dir=prometheus_dir,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_version=ssl_version,
        ssl_cert_reqs=ssl_cert_reqs,
        ssl_ca_certs=ssl_ca_certs,
        ssl_ciphers=ssl_ciphers,
        development_mode=development_mode,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
