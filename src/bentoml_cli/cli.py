from __future__ import annotations

import click
import psutil


def create_bentoml_cli() -> click.Command:
    from bentoml._internal.configuration import BENTOML_VERSION
    from bentoml._internal.context import server_context
    from bentoml_cli.bentos import bento_command
    from bentoml_cli.cloud import cloud_command
    from bentoml_cli.containerize import containerize_command
    from bentoml_cli.deployment import deploy_command
    from bentoml_cli.deployment import deployment_command
    from bentoml_cli.deployment import develop_command
    from bentoml_cli.env import env_command
    from bentoml_cli.models import model_command
    from bentoml_cli.secret import secret_command
    from bentoml_cli.serve import serve_command
    from bentoml_cli.start import start_command
    from bentoml_cli.utils import BentoMLCommandGroup
    from bentoml_cli.utils import get_entry_points

    server_context.service_type = "cli"

    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=BentoMLCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(BENTOML_VERSION, "-v", "--version")
    def bentoml_cli():
        """
        \b
        ██████╗ ███████╗███╗   ██╗████████╗ ██████╗ ███╗   ███╗██╗
        ██╔══██╗██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗████╗ ████║██║
        ██████╔╝█████╗  ██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║██║
        ██╔══██╗██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║██║
        ██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║███████╗
        ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝
        """

    # Add top-level CLI commands
    bentoml_cli.add_command(env_command)
    bentoml_cli.add_command(cloud_command)
    bentoml_cli.add_command(model_command)
    bentoml_cli.add_subcommands(bento_command)
    bentoml_cli.add_subcommands(start_command)
    bentoml_cli.add_subcommands(serve_command)
    bentoml_cli.add_command(containerize_command)
    bentoml_cli.add_command(deploy_command)
    bentoml_cli.add_command(develop_command)
    bentoml_cli.add_command(deployment_command)
    bentoml_cli.add_command(secret_command)
    # Load commands from extensions
    for ep in get_entry_points("bentoml.commands"):
        bentoml_cli.add_command(ep.load())

    if psutil.WINDOWS:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    return bentoml_cli


cli = create_bentoml_cli()


if __name__ == "__main__":
    cli()
