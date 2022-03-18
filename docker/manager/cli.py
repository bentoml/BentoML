import logging

import click
from manager import __version__ as MANAGER_VERSION

from .build import add_build_command
from .generate import add_generation_command
from .authenticate import add_authenticate_command
from ._internal.groups import ManagerCommandGroup

logger = logging.getLogger(__name__)


def create_manager_cli():

    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=ManagerCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(MANAGER_VERSION, "-v", "--version")
    def cli() -> None:
        """
        [bold yellow]Manager[/bold yellow]: BentoML's Docker Images release management system.

        \b
        [bold red]Features[/bold red]:
            :memo: Multiple Python version: 3.7, 3.8, 3.9+, ...
            :memo: Multiple platform: arm64v8, amd64, ppc64le, ...
            :memo: Multiple Linux Distros that you love: Debian, Ubuntu, UBI, alpine, ...

        Get started with:
            $ manager --help
        """

    add_build_command(cli)
    add_generation_command(cli)
    add_authenticate_command(cli)

    return cli


cli = create_manager_cli()

if __name__ == "__main__":
    cli()
