# type: ignore[reportUnusedFunction]
import sys
import typing as t
import logging

import click

from ..configuration.containers import DeploymentContainer

logger = logging.getLogger(__name__)

DEFAULT_DEV_SERVER_HOST = "127.0.0.1"
DEFAULT_RELAOD_DELAY = 0.25


def add_serve_command(cli: click.Group) -> None:
    @cli.command()
    @click.argument("bento", type=click.STRING)
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
        default=DeploymentContainer.service_port.get(),
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=None,
        help="The host to bind for the REST api server [defaults: 127.0.0.1(dev), 0.0.0.0(production)]",
        envvar="BENTOML_HOST",
    )
    @click.option(
        "--backlog",
        type=click.INT,
        default=DeploymentContainer.api_server_config.backlog.get(),
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
        "--reload-delay",
        type=click.FLOAT,
        help="Delay in seconds between each check if the Service needs to be reloaded",
        show_default=True,
        default=DEFAULT_RELAOD_DELAY,
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=".",
        show_default=True,
    )
    @click.option(
        "--run-with-ngrok",  # legacy option
        "--ngrok",
        is_flag=True,
        default=False,
        help="Use ngrok to relay traffic on a public endpoint to the local BentoServer, only available in dev mode",
        show_default=True,
    )
    def serve(
        bento: str,
        production: bool,
        port: int,
        host: t.Optional[str],
        backlog: int,
        reload: bool,
        reload_delay: float,
        working_dir: str,
        run_with_ngrok: bool,
    ) -> None:
        """Start BentoServer from BENTO

        BENTO is the serving target: it can be the import path of a bentoml.Service
        instance; a tag to a Bento in local Bento store; or a file path to a Bento
        directory, e.g.:

        \b
        Serve from a bentoml.Service instance source code(for development use only):
            bentoml serve fraud_detector.py:svc

        \b
        Serve from a Bento built in local store:
            bentoml serve fraud_detector:4tht2icroji6zput3suqi5nl2
            bentoml serve fraud_detector:latest

        \b
        Serve from a Bento directory:
            bentoml serve ./fraud_detector_bento
        """
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        if production:
            if run_with_ngrok:
                logger.warning(
                    "--run-with-ngrok option is not supported in production server"
                )
            if reload:
                logger.warning("--reload option is not supported in production server")

            from ..server import serve_production

            serve_production(
                bento,
                working_dir=working_dir,
                port=port,
                host=DeploymentContainer.service_host.get() if host is None else host,
                backlog=backlog,
            )
        else:
            from ..server import serve_development

            serve_development(
                bento,
                working_dir=working_dir,
                with_ngrok=run_with_ngrok,
                port=port,
                host=DEFAULT_DEV_SERVER_HOST if host is None else host,
                reload=reload,
                reload_delay=reload_delay,
            )
