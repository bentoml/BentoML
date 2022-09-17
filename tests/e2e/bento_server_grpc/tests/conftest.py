# pylint: disable=unused-argument
from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import psutil
import pytest

from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from contextlib import ExitStack

    from _pytest.nodes import Item as _PytestItem

    from bentoml._internal.server.metrics.prometheus import PrometheusClient

    # fixturenames and funcargs will be added dynamically
    # inside tests generation lifecycle
    class FunctionItem(_PytestItem):
        fixturenames: list[str]
        funcargs: dict[str, t.Any]


@pytest.fixture(scope="module", name="metrics_client")
def fixture_metrics_client() -> PrometheusClient:
    return BentoMLContainer.metrics_client.get()


@pytest.fixture(scope="module")
def host(
    bentoml_home: str,
    deployment_mode: t.Literal["docker", "distributed", "standalone"],
    clean_context: ExitStack,
) -> t.Generator[str, None, None]:
    from bentoml.testing.server import host_bento

    # import os
    # PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # config_file = os.path.join(PROJECT_DIR, "tracing.yml")
    if psutil.WINDOWS:
        pytest.skip("gRPC is not supported on Windows.")
    with host_bento(
        "service:svc",
        deployment_mode=deployment_mode,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
        # config_file=config_file,
        use_grpc=True,
    ) as _host:
        yield _host
