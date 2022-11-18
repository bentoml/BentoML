# pylint: disable=unused-argument
from __future__ import annotations

import os
import sys
import typing as t
import subprocess
from typing import TYPE_CHECKING

import psutil
import pytest

if TYPE_CHECKING:
    from contextlib import ExitStack

    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.config import Config


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pytest_collection_modifyitems(
    session: Session, config: Config, items: list[Item]
) -> None:
    subprocess.check_call(
        ["pip", "install", "-r", f"{os.path.join(PROJECT_DIR, 'requirements.txt')}"]
    )
    subprocess.check_call([sys.executable, f"{os.path.join(PROJECT_DIR, 'train.py')}"])


@pytest.mark.usefixtures("change_test_dir")
@pytest.fixture(scope="module")
def host(
    bentoml_home: str,
    deployment_mode: t.Literal["container", "distributed", "standalone"],
    clean_context: ExitStack,
) -> t.Generator[str, None, None]:
    from bentoml.testing.server import host_bento

    if psutil.WINDOWS:
        pytest.skip("gRPC is not supported on Windows.")
    with host_bento(
        "service:svc",
        deployment_mode=deployment_mode,
        project_path=PROJECT_DIR,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
        use_grpc=True,
    ) as _host:
        yield _host
