# pylint: disable=redefined-outer-name

import typing as t
import pathlib
import subprocess

import pytest

import bentoml
from bentoml.testing.server import host_bento
from bentoml.testing.server import bentoml_build


def pytest_configure(config):  # pylint: disable=unused-argument
    import os
    import sys

    subprocess.check_call([sys.executable, pathlib.Path("train.py").absolute()])


@pytest.fixture(scope="session")
def host() -> t.Generator[str, None, None]:
    with host_bento(bento="tensorflow_mnist_demo:latest") as host:
        yield host
