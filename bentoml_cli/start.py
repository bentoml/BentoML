from __future__ import annotations

import sys
import logging

import click

logger = logging.getLogger(__name__)


def add_start_command(cli: click.Group) -> None:

    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.configuration.containers import BentoMLContainer

    @cli.command(hidden=True)
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "--remote-runner",
        type=click.STRING,
        multiple=True,
        envvar="BENTOML_SERVE_RUNNER_MAP",
        help="JSON string of runners map",
    )
    @click.option(
        "--port",
        type=click.INT,
        default=BentoMLContainer.service_port.get(),
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=BentoMLContainer.service_host.get(),
        help="The host to bind for the REST api server [defaults: 127.0.0.1(dev), 0.0.0.0(production)]",
        envvar="BENTOML_HOST",
    )
    @click.option(
        "--backlog",
        type=click.INT,
        default=BentoMLContainer.api_server_config.backlog.get(),
        help="The maximum number of pending connections.",
        show_default=True,
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=".",
        show_default=True,
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
    def start_http_server(  # type: ignore (unused warning)
        bento: str,
        remote_runner: list[str] | None,
        port: int,
        host: str,
        backlog: int,
        working_dir: str,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        ssl_keyfile_password: str | None,
        ssl_version: int | None,
        ssl_cert_reqs: int | None,
        ssl_ca_certs: str | None,
        ssl_ciphers: str | None,
    ) -> None:
        configure_server_logging()
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        from bentoml.start import start_http_server

        runner_map = dict([s.split("=", maxsplit=2) for s in remote_runner or []])
        logger.info(" Using remote runners: %s", runner_map)
        start_http_server(
            bento,
            runner_map=runner_map,
            working_dir=working_dir,
            port=port,
            host=host,
            backlog=backlog,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_keyfile_password=ssl_keyfile_password,
            ssl_version=ssl_version,
            ssl_cert_reqs=ssl_cert_reqs,
            ssl_ca_certs=ssl_ca_certs,
            ssl_ciphers=ssl_ciphers,
        )

    @cli.command(hidden=True)
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "--runner-name",
        type=click.STRING,
        required=True,
        envvar="BENTOML_SERVE_RUNNER_NAME",
        help="specify the runner name to serve",
    )
    @click.option(
        "--port",
        type=click.INT,
        default=BentoMLContainer.service_port.get(),
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=BentoMLContainer.service_host.get(),
        help="The host to bind for the REST api server [defaults: 127.0.0.1(dev), 0.0.0.0(production)]",
        envvar="BENTOML_HOST",
    )
    @click.option(
        "--backlog",
        type=click.INT,
        default=BentoMLContainer.api_server_config.backlog.get(),
        help="The maximum number of pending connections.",
        show_default=True,
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=".",
        show_default=True,
    )
    def start_runner_server(  # type: ignore (unused warning)
        bento: str,
        runner_name: str,
        port: int,
        host: str,
        backlog: int,
        working_dir: str,
    ) -> None:
        configure_server_logging()
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        from bentoml.start import start_runner_server

        start_runner_server(
            bento,
            runner_name=runner_name,
            working_dir=working_dir,
            port=port,
            host=host,
            backlog=backlog,
        )
