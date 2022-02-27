import functools

import click

from bentoml import __version__ as BENTOML_VERSION

from .yatai import add_login_command
from .click_utils import BentoMLCommandGroup
from .bento_server import add_serve_command
from .containerize import add_containerize_command
from ..configuration import is_pypi_installed_bentoml
from .bento_management import add_bento_management_commands
from .model_management import add_model_management_commands


@functools.lru_cache(maxsize=1)
def _rich_callback():
    import sys

    if sys.stdout.isatty() and not is_pypi_installed_bentoml():
        # add traceback when in interactive shell for development
        from rich.traceback import install

        install(suppress=[click])


def create_bentoml_cli():
    _rich_callback()
    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=BentoMLCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(BENTOML_VERSION, "-v", "--version")  # type: ignore
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
