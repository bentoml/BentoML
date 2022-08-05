from __future__ import annotations

import click
import psutil

from bentoml import __version__ as BENTOML_VERSION
from bentoml_cli.env import add_env_command
from bentoml_cli.serve import add_serve_command
from bentoml_cli.utils import BentoMLCommandGroup
from bentoml_cli.yatai import add_login_command
from bentoml_cli.containerize import add_containerize_command
from bentoml_cli.bento_management import add_bento_management_commands
from bentoml_cli.model_management import add_model_management_commands


def create_bentoml_cli():

    from bentoml._internal.context import component_context

    component_context.component_name = "cli"

    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=BentoMLCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(BENTOML_VERSION, "-v", "--version")
    def cli():
        """BentoML CLI"""

    # Add top-level CLI commands
    add_env_command(cli)
    add_login_command(cli)
    add_bento_management_commands(cli)
    add_model_management_commands(cli)
    add_serve_command(cli)
    add_containerize_command(cli)

    if psutil.WINDOWS:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    return cli


cli = create_bentoml_cli()


if __name__ == "__main__":
    cli()
