from __future__ import annotations

import sys
import logging

import click

logger = logging.getLogger(__name__)

DEFAULT_DEV_SERVER_HOST = "127.0.0.1"


def add_serve_command(cli: click.Group) -> None:

    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.configuration.containers import BentoMLContainer

    @cli.command()
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "--component",
        type=click.STRING,
        default=None,
        envvar="BENTOML_SERVE_COMPONENT",
        help="[Experimental] Component (`api-server` or `runner`) to serve, if not specified, will serve all components",
        hidden=True,
    )
    @click.option(
        "--runner-name",
        type=click.STRING,
        default=None,
        envvar="BENTOML_SERVE_RUNNER_NAME",
        help="[Experimental] required if `component` is `runner`, specify the runner name to serve",
        hidden=True,
    )
    @click.option(
        "--remote-runner",
        type=click.STRING,
        multiple=True,
        envvar="BENTOML_SERVE_RUNNER_MAP",
        help="[Experimental] required if `component` is `api-server' JSON string of runners map",
        hidden=True,
    )
    @click.option(
        "--production",
        type=click.BOOL,
        help="Run the BentoServer in production mode",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "--port",
        type=click.INT,
        default=BentoMLContainer.service_port.get,
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=BentoMLContainer.service_host.get,
        help="The host to bind for the REST api server [defaults: 127.0.0.1(dev), 0.0.0.0(production)]",
        envvar="BENTOML_HOST",
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        default=None,
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="BENTOML_API_WORKERS",
    )
    @click.option(
        "--backlog",
        type=click.INT,
        default=BentoMLContainer.api_server_config.backlog.get,
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
    def serve(  # type: ignore (unused warning)
        bento: str,
        component: str | None,
        runner_name: str | None,
        remote_runner: list[str] | None,
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
        """Start a :code:`BentoServer` from a given ``BENTO`` 🍱

        ``BENTO`` is the serving target, it can be the import as:
            - the import path of a :code:`bentoml.Service` instance
            - a tag to a Bento in local Bento store
            - a folder containing a valid `bentofile.yaml` build file with a `service` field, which provides the import path of a :code:`bentoml.Service` instance
            - a path to a built Bento (for internal & debug use only)

        e.g.:

        \b
        Serve from a bentoml.Service instance source code (for development use only):
            :code:`bentoml serve fraud_detector.py:svc`

        \b
        Serve from a Bento built in local store:
            :code:`bentoml serve fraud_detector:4tht2icroji6zput3suqi5nl2`
            :code:`bentoml serve fraud_detector:latest`

        \b
        Serve from a Bento directory:
            :code:`bentoml serve ./fraud_detector_bento`

        \b
        If :code:`--reload` is provided, BentoML will detect code and model store changes during development, and restarts the service automatically.

            The `--reload` flag will:
            - be default, all file changes under `--working-dir` (default to current directory) will trigger a restart
            - when specified, respect :obj:`include` and :obj:`exclude` under :obj:`bentofile.yaml` as well as the :obj:`.bentoignore` file in `--working-dir`, for code and file changes
            - all model store changes will also trigger a restart (new model saved or existing model removed)
        """
        print(port)
        configure_server_logging()

        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        if component is None:
            # If no component is provided, serve the entire Bento.
            if production:
                if reload:
                    logger.warning(
                        "'--reload' is not supported with '--production'; ignoring"
                    )

                from bentoml.serve import serve_production

                serve_production(
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
                from bentoml.serve import serve_development

                serve_development(
                    bento,
                    working_dir=working_dir,
                    port=port,
                    host=DEFAULT_DEV_SERVER_HOST if host is None else host,
                    reload=reload,
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile,
                    ssl_keyfile_password=ssl_keyfile_password,
                    ssl_version=ssl_version,
                    ssl_cert_reqs=ssl_cert_reqs,
                    ssl_ca_certs=ssl_ca_certs,
                    ssl_ciphers=ssl_ciphers,
                )
            return

        if component.lower() == "runner":
            if runner_name is None:
                raise ValueError("--runner-name is required with '--component runner'")
            if reload:
                raise ValueError("--reload is not supported with --runner-name")
            if api_workers is not None:
                raise ValueError("--api-workers is not supported with --runner-name")
            if (
                ssl_certfile
                or ssl_keyfile
                or ssl_keyfile_password
                or ssl_version
                or ssl_cert_reqs
                or ssl_ca_certs
                or ssl_ciphers
            ):
                raise ValueError(
                    "--ssl-certfile, --ssl-keyfile, --ssl-keyfile-password, "
                    "--ssl-version, --ssl-cert-reqs, --ssl-ca-certs, --ssl-ciphers "
                    "are not supported with '--component runner'"
                )

            from bentoml.serve import serve_runner

            serve_runner(
                bento,
                runner_name=runner_name,
                working_dir=working_dir,
                port=port,
                host=host,
                backlog=backlog,
            )
            return

        elif component.lower() == "api-server":
            if reload:
                raise ValueError(
                    "--reload is not supported with '--component api-server'"
                )
            if runner_name is not None:
                raise ValueError(
                    "--runner-name is not needed with '--component api-server'"
                )

            runner_map = dict([s.split("=", maxsplit=2) for s in remote_runner or []])

            from bentoml.serve import serve_bare_api_server

            serve_bare_api_server(
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
            return

        else:
            raise ValueError(
                f"Invalid --component value: {component}; must be 'runner' or 'api-server'"
            )
