import logging

import click

from ..configuration.containers import BentoServerContainer

logger = logging.getLogger(__name__)


def add_serve_command(cli) -> None:
    @cli.command(
        help="Start a BentoServer serving a Service either imported from source code "
        "or loaded from Bento",
    )
    @click.argument("svc_import_path_or_bento_tag", type=click.STRING)
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="Look for Service in the specified directory",
        default="./",
    )
    @click.option(
        "--production",
        type=click.Path(),
        help="Run the BentoServer in production mode",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "--port",
        type=click.INT,
        default=BentoServerContainer.config.port.get(),
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--run-with-ngrok",  # legacy option
        "--ngrok",
        is_flag=True,
        default=False,
        help="Use ngrok to relay traffic on a public endpoint to the local BentoServer",
        envvar="BENTOML_ENABLE_NGROK",
        show_default=True,
    )
    def serve(
        svc_import_path_or_bento_tag,
        working_dir,
        port,
        run_with_ngrok,
        production,
    ) -> None:

        if production:
            from bentoml._internal.server import serve_production

            serve_production(
                svc_import_path_or_bento_tag,
                working_dir=working_dir,
                port=port,
            )
        else:
            from bentoml._internal.server import serve_development

            serve_development(
                svc_import_path_or_bento_tag,
                working_dir=working_dir,
                with_ngroxy=run_with_ngrok,
                port=port,
            )
