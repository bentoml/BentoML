from __future__ import annotations

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
        """Test given Bento by providing the bento_tag.

        \b
        `bentoml test <bento>` command will run the tests spegstcified in the `bentofile.yaml` file against the given Bento.
        For example: `bentoml test iris_classifier:v1.2.0`
        In the background, `bentoml test` command will create a docker image of the given bento,
        run a docker container from the docker image, and run the tests against the container.
        For every test case, the test result will be printed to the console.
        After the tests are finished, the docker image and the docker container will be removed.
        """
        test_bento_bundle(bento_tag)
