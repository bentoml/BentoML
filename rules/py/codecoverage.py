# type: ignore
"Stripped-down version of pytest-cov."

from __future__ import annotations

import os
import typing as t
from os.path import expandvars

import coverage
import coverage.files
import coverage.control
import coverage.collector

# Coverage likes to realpath filenames but we need paths relative to the exec
# root. So we patch it a bit get it inline.

# copy paste from coverage.files.abs_file
def abs_file(filename: str):
    """Return the absolute normalized form of `filename`."""
    path = expandvars(os.path.expanduser(filename))
    path = os.path.abspath(path)
    path = coverage.files.actual_path(path)
    return str(path)


coverage.collector.abs_file = abs_file
coverage.files.abs_file = abs_file
coverage.control.abs_file = abs_file


def pytest_configure(config: t.Any):
    if not os.getenv("COVERAGE_OUTPUT_FILE", None) or not os.getenv(
        "COVERAGE_MANIFEST", None
    ):
        # These variables are expected to be set by `bazel test` if coverage is enabled
        return

    # Read list of files to be covered from $COVERAGE_MANIFEST
    coverage_source = open(os.getenv("COVERAGE_MANIFEST")).read().split()  # type: ignore[arg-type]
    coverage_output = os.getenv("COVERAGE_OUTPUT_FILE")

    if coverage_source:
        plugin = CovPlugin(coverage_source, coverage_output)
        config.pluginmanager.register(plugin)


class CovPlugin(object):
    def __init__(
        self, coverage_source: t.List[str], coverage_output: t.Optional[str]
    ) -> None:
        self.cov = coverage.Coverage(
            source=[s.replace("/", ".")[:-3] for s in coverage_source]
        )
        self.cov.start()
        self.coverage_output = coverage_output

    def pytest_sessionfinish(self, session, exitstatus):
        self.cov.stop()
        self.cov.xml_report(outfile=self.coverage_output, ignore_errors=False)
