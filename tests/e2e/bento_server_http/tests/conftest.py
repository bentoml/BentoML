from __future__ import annotations

import os
import typing as t
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from contextlib import ExitStack

    from _pytest.fixtures import FixtureRequest as _PytestFixtureRequest

    class FixtureRequest(_PytestFixtureRequest):
        param: str


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(
    name="server_config_file",
    params=["default.yml", "cors_enabled.yml"],
    scope="session",
)
def fixture_server_config_file(request: FixtureRequest) -> str:
    return os.path.join(PROJECT_DIR, "configs", request.param)


@pytest.fixture(scope="module")
def host(
    bentoml_home: str,
    deployment_mode: t.Literal["docker", "distributed", "standalone"],
    server_config_file: str,
    clean_context: ExitStack,
) -> t.Generator[str, None, None]:
    from bentoml.testing.server import host_bento

    with host_bento(
        "service:svc",
        config_file=server_config_file,
        deployment_mode=deployment_mode,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
    ) as _host:
        yield _host
