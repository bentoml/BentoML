from __future__ import annotations

import os
import shlex
import time

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.xfail(
    os.getenv("GITHUB_ACTIONS") is not None, reason="Skip on distributed tests for now."
)
def test_regression(runner: CliRunner):
    """
    This test will determine if our CLI are running in an efficient manner.
    CLI runtime are ~ 340ms via loading entrypoint.
    The bulk of the time lies in how Python3 resolves dependencies and imports the package.
    The core runtime for loading bentoml library is around 170ms, and hence the threshold is set.
    This upper bound is loosely defined, but provide a good enough upper bound for the regression test.
    """
    from bentoml_cli.cli import cli

    # note that this should only be run in a single process.
    with runner.isolation():
        start = time.perf_counter_ns()
        ret = cli.main(
            args=shlex.split("--help"), prog_name="bentoml", standalone_mode=False
        )
        finish = time.perf_counter_ns() - start
    assert not ret and finish <= 1.7 * 1e6
