from __future__ import annotations

import os
import sys
import typing as t
import contextlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

import bentoml
from bentoml._internal.configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.config import Config
    from _pytest.tmpdir import TempPathFactory
    from _pytest.fixtures import FixtureRequest

PROJECT_DIR = Path(__file__).parent.parent


def pytest_collection_modifyitems(session: Session, config: Config, items: list[Item]):
    try:
        print(f"Found {bentoml.models.get('iris_clf')}, skipping model saving.")
    except bentoml.exceptions.NotFound:
        subprocess.check_call(
            [sys.executable, PROJECT_DIR.joinpath("train.py").__fspath__()]
        )


@pytest.fixture(
    name="monitoring_type", params=["default", "otlp"], scope="session", autouse=True
)
def fixture_monitoring_type(request: FixtureRequest) -> str:
    BentoMLContainer.config.monitoring.type.set(request.param)
    return request.param


@pytest.fixture(name="monitoring_dir", scope="session")
def fixture_monitoring_dir(tmp_path_factory: TempPathFactory) -> str:
    d = tmp_path_factory.mktemp("monitoring")
    os.environ["MONITORING_LOG_PATH"] = d.__fspath__()
    return d


@pytest.fixture(scope="session")
def host(
    bentoml_home: str,
    deployment_mode: t.Literal["container", "distributed", "standalone"],
    clean_context: contextlib.ExitStack,
    monitoring_dir: Path,
):
    from bentoml.testing.server import host_bento

    with host_bento(
        "service:svc",
        project_path=PROJECT_DIR.__fspath__(),
        deployment_mode=deployment_mode,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
    ) as _host:
        client = bentoml.client.Client.from_url(_host)
        for _ in range(10):
            client.classify(np.array([4.9, 3.0, 1.4, 0.2]))
