import click

from bentoml import __version__

from .yatai import add_login_command
from .click_utils import BentoMLCommandGroup
from .bento_server import add_serve_command
from .containerize import add_containerize_command
from .bento_management import add_bento_management_commands
from .model_management import add_model_management_commands


def create_bentoml_cli():
    @click.group(cls=BentoMLCommandGroup)
    @click.version_option(version=__version__)
    def cli():
        """BentoML CLI"""

    # Add top-level CLI commands
    add_login_command(cli)
    add_bento_management_commands(cli)
    add_model_management_commands(cli)
    add_serve_command(cli)
    add_containerize_command(cli)

    return cli


cli = create_bentoml_cli()


if __name__ == "__main__":
    cli()
