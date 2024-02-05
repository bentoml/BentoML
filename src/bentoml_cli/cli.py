from __future__ import annotations

import importlib.metadata

import click
import psutil


def create_bentoml_cli() -> click.Command:
    from bentoml._internal.context import component_context
    from bentoml_cli.bentos import bentos
    from bentoml_cli.cloud import cloud_command
    from bentoml_cli.containerize import containerize_command
    from bentoml_cli.deployment import deploy_command
    from bentoml_cli.deployment import deployment_command
    from bentoml_cli.env import env_command
    from bentoml_cli.models import model_command
    from bentoml_cli.serve import serve_command
    from bentoml_cli.start import start_command
    from bentoml_cli.utils import BentoMLCommandGroup

    component_context.component_type = "cli"

    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=BentoMLCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(importlib.metadata.version("bentoml"), "-v", "--version")
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
    bentoml_cli.add_subcommands(bentos)
    bentoml_cli.add_subcommands(start_command)
    bentoml_cli.add_subcommands(serve_command)
    bentoml_cli.add_command(containerize_command)
    bentoml_cli.add_command(deploy_command)
    bentoml_cli.add_command(deployment_command)

    if psutil.WINDOWS:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    return bentoml_cli


cli = create_bentoml_cli()


if __name__ == "__main__":
    cli()
