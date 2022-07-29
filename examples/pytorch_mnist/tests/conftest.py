# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import typing as t

import numpy as np
import pytest
from bentoml.testing.server import run_api_server


def pytest_configure(config):  # pylint: disable=unused-argument
    import os
    import subprocess
    import sys

    cmd = f"{sys.executable} {os.path.join(os.getcwd(), 'train.py')} --k-folds=0"
    subprocess.run(cmd, shell=True, check=True)


@pytest.fixture(scope="session")
def host() -> t.Generator[str, None, None]:
    import bentoml

    bentoml.build("service:svc")

    with run_api_server(
        bento="pytorch_mnist_demo:latest",
    ) as host:
        yield host
