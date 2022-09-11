# pylint: disable=unused-argument
from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import psutil
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.globals_test import reset_trace_globals
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from contextlib import ExitStack

    from _pytest.nodes import Item as _PytestItem
    from _pytest.config import Config

    from bentoml._internal.server.metrics.prometheus import PrometheusClient

    # fixturenames and funcargs will be added dynamically
    # inside tests generation lifecycle
    class FunctionItem(_PytestItem):
        fixturenames: list[str]
        funcargs: dict[str, t.Any]


def create_tracer_provider(
    **kwargs: t.Any,
) -> tuple[TracerProvider, InMemorySpanExporter]:
    tracer_provider = TracerProvider(**kwargs)
    memory_exporter = InMemorySpanExporter()
    span_processor = export.SimpleSpanProcessor(memory_exporter)
    tracer_provider.add_span_processor(span_processor)
    return tracer_provider, memory_exporter


OTEL_MARKER = "otel"
SKIP_DEPLOYMENT = "skip_deployment_mode"


def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{OTEL_MARKER}: mark the test to use OpenTelemetry fixtures.",
    )


def pytest_runtest_setup(item: FunctionItem):
    marker = item.get_closest_marker(OTEL_MARKER)
    if marker:
        tracer_provider, memory_exporter = create_tracer_provider()
        BentoMLContainer.tracer_provider.set(tracer_provider)
        # This is done because set_tracer_provider cannot override the
        # current tracer provider.
        reset_trace_globals()
        trace_api.set_tracer_provider(tracer_provider)
        memory_exporter.clear()
        # handling fixtures
        fixturenames: list[str] = item.fixturenames
        funcargs = item.funcargs
        if "tracer_provider" in fixturenames:
            fixturenames.remove("tracer_provider")
        fixturenames.insert(0, "tracer_provider")
        funcargs["tracer_provider"] = tracer_provider
        if "memory_exporter" in fixturenames:
            fixturenames.remove("memory_exporter")
        fixturenames.insert(0, "memory_exporter")
        funcargs["memory_exporter"] = memory_exporter


def pytest_runtest_teardown(item: FunctionItem, nextitem: FunctionItem | None):
    if item.get_closest_marker(OTEL_MARKER):
        reset_trace_globals()
        BentoMLContainer.tracer_provider.reset()


@pytest.fixture(scope="module", name="metrics_client")
def fixture_metrics_client() -> PrometheusClient:
    return BentoMLContainer.metrics_client.get()


@pytest.fixture(scope="module")
def host(
    bentoml_home: str,
    deployment_mode: str,
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
