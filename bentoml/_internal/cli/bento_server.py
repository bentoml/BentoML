import logging

import click

from ..configuration.containers import BentoServerContainer

logger = logging.getLogger(__name__)


def add_serve_command(cli):
    @cli.command(
        help="Start a BentoServer serving a Service either imported from source code "
        "or loaded from Bento",
    )
    @click.argument("svc_import_path_or_bento_tag", type=click.STRING)
    @click.option(
        "--working-dir",
        type=click.Path,
        help="Look for Service in the specified directory",
    )
    @click.option(
        "--production",
        type=click.Path,
        help="Run the BentoServer in production mode",
        flag=True,
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
        "--ngrok",
        "--run-with-ngrok",  # legacy option
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
    ):
        from ..service.loader import load

        svc = load(svc_import_path_or_bento_tag, working_dir)

        if production:
            ...
        else:
            ...
