from __future__ import annotations

import json

import click
import psutil

from bentoml import __version__ as BENTOML_VERSION

from .yatai import add_login_command
from ..utils import console
from .click_utils import BentoMLCommandGroup
from .bento_server import add_serve_command
from .containerize import add_containerize_command
from .bento_management import add_bento_management_commands
from .model_management import add_model_management_commands


def create_bentoml_cli():
    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=BentoMLCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(BENTOML_VERSION, "-v", "--version")  # type: ignore
    def cli():
        """BentoML CLI"""

    @cli.command()
    def env() -> None:  # type: ignore # noqa
        """Provide BentoML's environment information. Mainly used for debugging purposes and issues tracking."""
        from platform import platform
        from platform import python_version

        info = {
            "Python version": python_version(),
            "BentoML version": BENTOML_VERSION,
            "Platform info": platform(),
        }
        console.print_json(json.dumps(info, indent=2))

    # Add top-level CLI commands
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
