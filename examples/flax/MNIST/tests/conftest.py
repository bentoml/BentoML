from __future__ import annotations

import os
import sys
import typing as t
import contextlib
import subprocess
from typing import TYPE_CHECKING

import psutil
import pytest

import bentoml
from bentoml.testing.server import host_bento
from bentoml._internal.configuration.containers import BentoMLContainer

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
    try:
        m = bentoml.models.get("mnist_flax")
        print(f"Model exists: {m}")
    except bentoml.exceptions.NotFound:
        subprocess.check_call(
            [
                sys.executable,
                f"{os.path.join(PROJECT_DIR, 'train.py')}",
                "--num-epochs",
                "2",  # 2 epochs for faster testing
                "--lr",
                "0.22",  # speed up training time
                "--enable-tensorboard",
            ]
        )


@pytest.fixture(name="enable_grpc", params=[True, False], scope="session")
def fixture_enable_grpc(request: FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session", autouse=True)
def clean_context() -> t.Generator[contextlib.ExitStack, None, None]:
    stack = contextlib.ExitStack()
    yield stack
    stack.close()


@pytest.fixture(
    name="deployment_mode",
    params=["container", "distributed", "standalone"],
    scope="session",
)
def fixture_deployment_mode(request: FixtureRequest) -> str:
    return request.param


@pytest.mark.usefixtures("change_test_dir")
@pytest.fixture(scope="module")
def host(
    deployment_mode: t.Literal["container", "distributed", "standalone"],
    clean_context: ExitStack,
    enable_grpc: bool,
) -> t.Generator[str, None, None]:
    if enable_grpc and psutil.WINDOWS:
        pytest.skip("gRPC is not supported on Windows.")

    with host_bento(
        "service:svc",
        deployment_mode=deployment_mode,
        project_path=PROJECT_DIR,
        bentoml_home=BentoMLContainer.bentoml_home.get(),
        clean_context=clean_context,
        use_grpc=enable_grpc,
    ) as _host:
        yield _host
