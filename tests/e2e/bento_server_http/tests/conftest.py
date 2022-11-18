# pylint: disable=unused-argument
from __future__ import annotations

import os
import sys
import typing as t
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from contextlib import ExitStack

    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.config import Config
    from _pytest.fixtures import FixtureRequest as _PytestFixtureRequest

    class FixtureRequest(_PytestFixtureRequest):
        param: str


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pytest_collection_modifyitems(
    session: Session, config: Config, items: list[Item]
) -> None:
    subprocess.check_call(
        ["pip", "install", "-r", f"{os.path.join(PROJECT_DIR, 'requirements.txt')}"]
    )
    subprocess.check_call([sys.executable, f"{os.path.join(PROJECT_DIR, 'train.py')}"])


@pytest.fixture(
    name="server_config_file",
    params=["default.yml", "cors_enabled.yml"],
    scope="session",
)
def fixture_server_config_file(request: FixtureRequest) -> str:
    return os.path.join(PROJECT_DIR, "configs", request.param)


@pytest.mark.usefixtures("change_test_dir")
@pytest.fixture(scope="session")
def host(
    bentoml_home: str,
    deployment_mode: t.Literal["container", "distributed", "standalone"],
    server_config_file: str,
    clean_context: ExitStack,
) -> t.Generator[str, None, None]:
    from bentoml.testing.server import host_bento

    with host_bento(
        "service:svc",
        config_file=server_config_file,
        project_path=PROJECT_DIR,
        deployment_mode=deployment_mode,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
    ) as _host:
        yield _host
