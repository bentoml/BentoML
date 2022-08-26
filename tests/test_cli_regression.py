from __future__ import annotations

import time
import shlex

from click.testing import CliRunner

runner = CliRunner()


def test_regression():
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
        prog_name = runner.get_default_prog_name(cli)
        start = time.perf_counter_ns()
        try:
            _ = cli.main(args=shlex.split("--help"), prog_name=prog_name)
        except SystemExit:
            finish = time.perf_counter_ns() - start

            threshold = 1.7 * 1e6
            assert finish <= threshold
