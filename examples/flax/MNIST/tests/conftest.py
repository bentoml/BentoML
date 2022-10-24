from __future__ import annotations

import os
import sys
import typing as t
import subprocess
from typing import TYPE_CHECKING

import psutil
import pytest

from bentoml.testing.server import host_bento

if TYPE_CHECKING:
    from contextlib import ExitStack

    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.config import Config
    from _pytest.fixtures import FixtureRequest as _PytestFixtureRequest

    class FixtureRequest(_PytestFixtureRequest):
        param: str


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config: Config):  # pylint: disable=unused-argument
    if "PYTEST_PLUGINS" not in os.environ:
        os.environ["PYTEST_PLUGINS"] = "bentoml.testing.pytest_plugin"


def pytest_unconfigure(config: Config):  # pylint: disable=unused-argument
    if "PYTEST_PLUGINS" in os.environ:
        del os.environ["PYTEST_PLUGINS"]


def pytest_collection_modifyitems(
    session: Session, config: Config, items: list[Item]
) -> None:
    subprocess.check_call(
        [
            sys.executable,
            f"{os.path.join(PROJECT_DIR, 'train.py')}",
            "--num-epochs",
            "2",
        ]
    )


@pytest.fixture(name="enable_grpc", params=[True, False], scope="session")
def fixture_enable_grpc(request: FixtureRequest) -> str:
    return request.param


@pytest.mark.usefixtures("change_test_dir")
@pytest.fixture(scope="module")
def host(
    bentoml_home: str,
    deployment_mode: t.Literal["docker", "distributed", "standalone"],
    clean_context: ExitStack,
    enable_grpc: bool,
) -> t.Generator[str, None, None]:
    if enable_grpc and psutil.WINDOWS:
        pytest.skip("gRPC is not supported on Windows.")

    with host_bento(
        "service:svc",
        deployment_mode=deployment_mode,
        project_path=PROJECT_DIR,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
        use_grpc=enable_grpc,
    ) as _host:
        yield _host
