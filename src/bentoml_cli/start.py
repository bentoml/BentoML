from __future__ import annotations

import os
import sys
import json
import logging
from urllib.parse import urlparse

import click

logger = logging.getLogger(__name__)


def add_start_command(cli: click.Group) -> None:
    from bentoml.grpc.utils import LATEST_PROTOCOL_VERSION
    from bentoml._internal.utils import add_experimental_docstring
    from bentoml._internal.configuration.containers import BentoMLContainer

    @cli.command(hidden=True)
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "--remote-runner",
        type=click.STRING,
        multiple=True,
        envvar="BENTOML_SERVE_REMOTE_RUNNER",
        help="list of runners map",
    )
    @click.option(
        "--runner-map",
        type=click.STRING,
        envvar="BENTOML_SERVE_RUNNER_MAP",
        help="[Deprecated] use --remote-runner instead. "
        "JSON string of runners map. For backword compatibility for yatai < 1.0.0",
    )
    @click.option(
        "--bind",
        type=click.STRING,
        help="[Deprecated] use --host and --port instead."
        "Bind address for the server. For backword compatibility for yatai < 1.0.0",
        required=False,
    )
    @click.option(
        "--port",
        type=click.INT,
        default=BentoMLContainer.http.port.get(),
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=BentoMLContainer.http.host.get(),
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
        "--api-workers",
        type=click.INT,
        default=BentoMLContainer.api_server_workers.get(),
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="BENTOML_API_WORKERS",
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=None,
        show_default=True,
    )
    @click.option(
        "--ssl-certfile",
        type=str,
        default=BentoMLContainer.ssl.certfile.get(),
        help="SSL certificate file",
    )
    @click.option(
        "--ssl-keyfile",
        type=str,
        default=BentoMLContainer.ssl.keyfile.get(),
        help="SSL key file",
    )
    @click.option(
        "--ssl-keyfile-password",
        type=str,
        default=BentoMLContainer.ssl.keyfile_password.get(),
        help="SSL keyfile password",
    )
    @click.option(
        "--ssl-version",
        type=int,
        default=BentoMLContainer.ssl.version.get(),
        help="SSL version to use (see stdlib 'ssl' module)",
    )
    @click.option(
        "--ssl-cert-reqs",
        type=int,
        default=BentoMLContainer.ssl.cert_reqs.get(),
        help="Whether client certificate is required (see stdlib 'ssl' module)",
    )
    @click.option(
        "--ssl-ca-certs",
        type=str,
        default=BentoMLContainer.ssl.ca_certs.get(),
        help="CA certificates file",
    )
    @click.option(
        "--ssl-ciphers",
        type=str,
        default=BentoMLContainer.ssl.ciphers.get(),
        help="Ciphers to use (see stdlib 'ssl' module)",
    )
    @add_experimental_docstring
    def start_http_server(  # type: ignore (unused warning)
        bento: str,
        remote_runner: list[str] | None,
        runner_map: str | None,
        bind: str | None,
        port: int,
        host: str,
        backlog: int,
        working_dir: str,
        api_workers: int | None,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        ssl_keyfile_password: str | None,
        ssl_version: int | None,
        ssl_cert_reqs: int | None,
        ssl_ca_certs: str | None,
        ssl_ciphers: str | None,
    ) -> None:
        """
        Start a HTTP API server standalone. This will be used inside Yatai.
        """
        if working_dir is None:
            if os.path.isdir(os.path.expanduser(bento)):
                working_dir = os.path.expanduser(bento)
            else:
                working_dir = "."
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        from bentoml.start import start_http_server

        if remote_runner:
            runner_map_dict = dict(
                [s.split("=", maxsplit=2) for s in remote_runner or []]
            )
        elif runner_map:
            runner_map_dict = json.loads(runner_map)
        else:
            runner_map_dict = {}

        if bind is not None:
            parsed = urlparse(bind)
            assert parsed.scheme == "tcp"
            host = parsed.hostname or host
            port = parsed.port or port

        click.echo(f"Using remote runners: {runner_map}")
        start_http_server(
            bento,
            runner_map=runner_map_dict,
            working_dir=working_dir,
            port=port,
            host=host,
            backlog=backlog,
            api_workers=api_workers,
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
        "--bind",
        type=click.STRING,
        help="[Deprecated] use --host and --port instead."
        "Bind address for the server. For backword compatibility for yatai < 1.0.0",
        required=False,
    )
    @click.option(
        "--port",
        type=click.INT,
        default=BentoMLContainer.http.port.get(),
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=BentoMLContainer.http.host.get(),
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
        default=None,
        show_default=True,
    )
    @add_experimental_docstring
    def start_runner_server(  # type: ignore (unused warning)
        bento: str,
        runner_name: str,
        bind: str | None,
        port: int,
        host: str,
        backlog: int,
        working_dir: str,
    ) -> None:
        """
        Start Runner server standalone. This will be used inside Yatai.
        """
        if working_dir is None:
            if os.path.isdir(os.path.expanduser(bento)):
                working_dir = os.path.expanduser(bento)
            else:
                working_dir = "."
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        from bentoml.start import start_runner_server

        if bind is not None:
            parsed = urlparse(bind)
            assert parsed.scheme == "tcp"
            host = parsed.hostname or host
            port = parsed.port or port

        start_runner_server(
            bento,
            runner_name=runner_name,
            working_dir=working_dir,
            port=port,
            host=host,
            backlog=backlog,
        )

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
        default=BentoMLContainer.grpc.port.get(),
        help="The port to listen on for the gRPC server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=BentoMLContainer.grpc.host.get(),
        help="The host to bind for the gRPC server (defaults: 0.0.0.0)",
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
        default=None,
        show_default=True,
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        default=BentoMLContainer.api_server_workers.get(),
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="BENTOML_API_WORKERS",
    )
    @click.option(
        "--enable-reflection",
        is_flag=True,
        default=BentoMLContainer.grpc.reflection.enabled.get(),
        type=click.BOOL,
        help="Enable reflection.",
    )
    @click.option(
        "--enable-channelz",
        is_flag=True,
        default=BentoMLContainer.grpc.channelz.enabled.get(),
        type=click.BOOL,
        help="Enable Channelz. See https://github.com/grpc/proposal/blob/master/A14-channelz.md.",
    )
    @click.option(
        "--max-concurrent-streams",
        default=BentoMLContainer.grpc.max_concurrent_streams.get(),
        type=click.INT,
        help="Maximum number of concurrent incoming streams to allow on a http2 connection.",
    )
    @click.option(
        "--ssl-certfile",
        type=str,
        default=BentoMLContainer.ssl.certfile.get(),
        help="SSL certificate file",
    )
    @click.option(
        "--ssl-keyfile",
        type=str,
        default=BentoMLContainer.ssl.keyfile.get(),
        help="SSL key file",
    )
    @click.option(
        "--ssl-ca-certs",
        type=str,
        default=BentoMLContainer.ssl.ca_certs.get(),
        help="CA certificates file",
    )
    @click.option(
        "-pv",
        "--protocol-version",
        type=click.Choice(["v1", "v1alpha1"]),
        help="Determine the version of generated gRPC stubs to use.",
        default=LATEST_PROTOCOL_VERSION,
        show_default=True,
    )
    @add_experimental_docstring
    def start_grpc_server(  # type: ignore (unused warning)
        bento: str,
        remote_runner: list[str] | None,
        port: int,
        host: str,
        backlog: int,
        api_workers: int | None,
        working_dir: str,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        enable_reflection: bool,
        ssl_ca_certs: str | None,
        enable_channelz: bool,
        max_concurrent_streams: int | None,
        protocol_version: str,
    ) -> None:
        """
        Start a gRPC API server standalone. This will be used inside Yatai.
        """
        if working_dir is None:
            if os.path.isdir(os.path.expanduser(bento)):
                working_dir = os.path.expanduser(bento)
            else:
                working_dir = "."
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        from bentoml.start import start_grpc_server

        runner_map = dict([s.split("=", maxsplit=2) for s in remote_runner or []])
        click.echo(f"Using remote runners: {runner_map}")
        start_grpc_server(
            bento,
            runner_map=runner_map,
            working_dir=working_dir,
            port=port,
            host=host,
            backlog=backlog,
            api_workers=api_workers,
            reflection=enable_reflection,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_ca_certs=ssl_ca_certs,
            channelz=enable_channelz,
            max_concurrent_streams=max_concurrent_streams,
            protocol_version=protocol_version,
        )
