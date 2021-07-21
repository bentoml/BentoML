import logging

import click

from .bento_management import add_bento_management_sub_commands
from .label import add_label_sub_commands
from .server import add_yatai_service_sub_commands

logger = logging.getLogger(__name__)


def create_yatai_cli_group():
    @click.group()
    def yatai_cli():
        """
        YATAI CLI Tool
        """

    return _cli


def create_yatai_cli():
    _cli = create_yatai_cli_group()

    add_bento_management_sub_commands(_cli)
    add_yatai_service_sub_commands(_cli)
    add_label_sub_commands(_cli)

    return _cli


_cli = create_yatai_cli()


if __name__ == "__main__":
    _cli()
