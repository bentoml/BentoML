import logging
import pytest
from click.testing import CliRunner
from bentoml.cli.bento_service import create_bento_service_cli

logger = logging.getLogger('bentoml.test')


def test_containerize(basic_bentoservice_v1):
    runner = CliRunner()

    cli = create_bento_service_cli()
    run_cmd = cli.commands["containerize"]
    result = runner.invoke(
        run_cmd,
        [
            basic_bentoservice_v1,
        ],
    )

    assert result.exit_code == 0
