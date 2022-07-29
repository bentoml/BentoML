# pylint: disable=redefined-outer-name

import pathlib
import subprocess
import typing as t

import bentoml
import pytest
from bentoml.testing.server import bentoml_build, host_bento


def pytest_configure(config):  # pylint: disable=unused-argument
    import os
    import sys

    subprocess.check_call([sys.executable, pathlib.Path("train.py").absolute()])


@pytest.fixture(scope="session")
def host() -> t.Generator[str, None, None]:
    with host_bento(bento="tensorflow_mnist_demo:latest") as host:
        yield host
