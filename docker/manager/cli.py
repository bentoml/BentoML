import logging

import click
from manager import __version__ as MANAGER_VERSION
from manager.build import add_build_command
from manager.tests import add_tests_command
from manager._utils import graceful_exit
from manager.generate import add_generation_command
from manager._click_utils import ManagerCommandGroup
from manager.authenticate import add_authenticate_command

logger = logging.getLogger(__name__)


@graceful_exit
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
    add_tests_command(cli)
    add_generation_command(cli)
    add_authenticate_command(cli)

    return cli


cli = create_manager_cli()

if __name__ == "__main__":
    cli()
