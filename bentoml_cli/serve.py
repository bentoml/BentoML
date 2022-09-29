from __future__ import annotations

import sys
import logging

import click

logger = logging.getLogger(__name__)

DEFAULT_DEV_SERVER_HOST = "127.0.0.1"


def add_serve_command(cli: click.Group) -> None:

    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.configuration.containers import BentoMLContainer

    @cli.command(aliases=["serve-http"])
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "--production",
        type=click.BOOL,
        help="Run the BentoServer in production mode",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "-p",
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
        help="The host to bind for the REST api server",
        envvar="BENTOML_HOST",
        show_default=True,
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        default=BentoMLContainer.api_server_workers.get(),
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="BENTOML_API_WORKERS",
        show_default=True,
    )
    @click.option(
        "--backlog",
        type=click.INT,
        default=BentoMLContainer.api_server_config.backlog.get(),
        help="The maximum number of pending connections.",
        show_default=True,
    )
    @click.option(
        "--reload",
        type=click.BOOL,
        is_flag=True,
        help="Reload Service when code changes detected, this is only available in development mode",
        default=False,
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
        show_default=True,
    )
    @click.option(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="SSL key file",
        show_default=True,
    )
    @click.option(
        "--ssl-keyfile-password",
        type=str,
        default=None,
        help="SSL keyfile password",
        show_default=True,
    )
    @click.option(
        "--ssl-version",
        type=int,
        default=None,
        help="SSL version to use (see stdlib 'ssl' module)",
        show_default=True,
    )
    @click.option(
        "--ssl-cert-reqs",
        type=int,
        default=None,
        help="Whether client certificate is required (see stdlib 'ssl' module)",
        show_default=True,
    )
    @click.option(
        "--ssl-ca-certs",
        type=str,
        default=None,
        help="CA certificates file",
        show_default=True,
    )
    @click.option(
        "--ssl-ciphers",
        type=str,
        default=None,
        help="Ciphers to use (see stdlib 'ssl' module)",
        show_default=True,
    )
    def serve(  # type: ignore (unused warning)
        bento: str,
        production: bool,
        port: int,
        host: str,
        api_workers: int | None,
        backlog: int,
        reload: bool,
        working_dir: str,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        ssl_keyfile_password: str | None,
        ssl_version: int | None,
        ssl_cert_reqs: int | None,
        ssl_ca_certs: str | None,
        ssl_ciphers: str | None,
    ) -> None:
        """Start a HTTP BentoServer from a given 🍱

        \b
        BENTO is the serving target, it can be the import as:
        - the import path of a 'bentoml.Service' instance
        - a tag to a Bento in local Bento store
        - a folder containing a valid 'bentofile.yaml' build file with a 'service' field, which provides the import path of a 'bentoml.Service' instance
        - a path to a built Bento (for internal & debug use only)

        e.g.:

        \b
        Serve from a bentoml.Service instance source code (for development use only):
            'bentoml serve fraud_detector.py:svc'

        \b
        Serve from a Bento built in local store:
            'bentoml serve fraud_detector:4tht2icroji6zput3suqi5nl2'
            'bentoml serve fraud_detector:latest'

        \b
        Serve from a Bento directory:
            'bentoml serve ./fraud_detector_bento'

        \b
        If '--reload' is provided, BentoML will detect code and model store changes during development, and restarts the service automatically.

        \b
        The '--reload' flag will:
        - be default, all file changes under '--working-dir' (default to current directory) will trigger a restart
        - when specified, respect 'include' and 'exclude' under 'bentofile.yaml' as well as the '.bentoignore' file in '--working-dir', for code and file changes
        - all model store changes will also trigger a restart (new model saved or existing model removed)
        """
        configure_server_logging()

        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        if production:
            if reload:
                logger.warning(
                    "'--reload' is not supported with '--production'; ignoring"
                )

            from bentoml.serve import serve_http_production

            serve_http_production(
                bento,
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
        else:
            from bentoml.serve import serve_http_development

            serve_http_development(
                bento,
                working_dir=working_dir,
                port=port,
                host=DEFAULT_DEV_SERVER_HOST if not host else host,
                reload=reload,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                ssl_keyfile_password=ssl_keyfile_password,
                ssl_version=ssl_version,
                ssl_cert_reqs=ssl_cert_reqs,
                ssl_ca_certs=ssl_ca_certs,
                ssl_ciphers=ssl_ciphers,
            )

    from bentoml._internal.utils import add_experimental_docstring

    @cli.command(name="serve-grpc")
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "--production",
        type=click.BOOL,
        help="Run the BentoServer in production mode",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "-p",
        "--port",
        type=click.INT,
        default=BentoMLContainer.grpc.port.get(),
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=BentoMLContainer.grpc.host.get(),
        help="The host to bind for the gRPC server",
        envvar="BENTOML_HOST",
        show_default=True,
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        default=BentoMLContainer.api_server_workers.get(),
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="BENTOML_API_WORKERS",
        show_default=True,
    )
    @click.option(
        "--reload",
        type=click.BOOL,
        is_flag=True,
        help="Reload Service when code changes detected, this is only available in development mode",
        default=False,
        show_default=True,
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
        "--enable-reflection",
        is_flag=True,
        default=BentoMLContainer.grpc.reflection.enabled.get(),
        type=click.BOOL,
        help="Enable reflection.",
        show_default=True,
    )
    @click.option(
        "--max-concurrent-streams",
        default=BentoMLContainer.grpc.max_concurrent_streams.get(),
        type=click.INT,
        help="Maximum number of concurrent incoming streams to allow on a http2 connection.",
        show_default=True,
    )
    @add_experimental_docstring
    def serve_grpc(  # type: ignore (unused warning)
        bento: str,
        production: bool,
        port: int,
        host: str,
        api_workers: int | None,
        backlog: int,
        reload: bool,
        working_dir: str,
        enable_reflection: bool,
        max_concurrent_streams: int | None,
    ):
        """Start a gRPC BentoServer from a given 🍱

        \b
        BENTO is the serving target, it can be the import as:
        - the import path of a 'bentoml.Service' instance
        - a tag to a Bento in local Bento store
        - a folder containing a valid 'bentofile.yaml' build file with a 'service' field, which provides the import path of a 'bentoml.Service' instance
        - a path to a built Bento (for internal & debug use only)

        e.g.:

        \b
        Serve from a bentoml.Service instance source code (for development use only):
            'bentoml serve-grpc fraud_detector.py:svc'

        \b
        Serve from a Bento built in local store:
            'bentoml serve-grpc fraud_detector:4tht2icroji6zput3suqi5nl2'
            'bentoml serve-grpc fraud_detector:latest'

        \b
        Serve from a Bento directory:
            'bentoml serve-grpc ./fraud_detector_bento'

        If '--reload' is provided, BentoML will detect code and model store changes during development, and restarts the service automatically.

        \b
        The '--reload' flag will:
        - be default, all file changes under '--working-dir' (default to current directory) will trigger a restart
        - when specified, respect 'include' and 'exclude' under 'bentofile.yaml' as well as the '.bentoignore' file in '--working-dir', for code and file changes
        - all model store changes will also trigger a restart (new model saved or existing model removed)
        """
        configure_server_logging()
        if production:
            if reload:
                logger.warning(
                    "'--reload' is not supported with '--production'; ignoring"
                )

            from bentoml.serve import serve_grpc_production

            serve_grpc_production(
                bento,
                working_dir=working_dir,
                port=port,
                host=host,
                backlog=backlog,
                api_workers=api_workers,
                max_concurrent_streams=max_concurrent_streams,
                reflection=enable_reflection,
            )
        else:
            from bentoml.serve import serve_grpc_development

            serve_grpc_development(
                bento,
                working_dir=working_dir,
                port=port,
                backlog=backlog,
                reload=reload,
                host=DEFAULT_DEV_SERVER_HOST if not host else host,
                max_concurrent_streams=max_concurrent_streams,
                reflection=enable_reflection,
            )
