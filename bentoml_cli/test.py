from __future__ import annotations

import sys
import typing as t
import logging
import click

from bentoml.bentos import test_bento_bundle

logger = logging.getLogger("bentoml")


def add_test_command(cli: click.Group) -> None:
    @cli.command()
    @click.argument("bento_tag", type=click.STRING)
    def test(
            bento_tag: str,
    ) -> None:
        test_bento_bundle(bento_tag)
