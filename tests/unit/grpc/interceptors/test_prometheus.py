from __future__ import annotations

import typing as t
import tempfile
from typing import TYPE_CHECKING
from asyncio import Future
from unittest.mock import MagicMock

import pytest

from bentoml.testing.grpc import create_channel
from bentoml.testing.grpc import async_client_call
from bentoml.testing.grpc import create_bento_servicer
from bentoml.testing.grpc import make_standalone_server

if TYPE_CHECKING:
    import grpc
    from _pytest.python import Metafunc
    from google.protobuf import wrappers_pb2

    from bentoml import Service
    from bentoml.grpc.v1alpha1 import service_pb2_grpc as services
    from bentoml.grpc.v1alpha1 import service_test_pb2 as pb_test
    from bentoml.grpc.interceptors.prometheus import PrometheusServerInterceptor
    from bentoml._internal.server.metrics.prometheus import PrometheusClient
else:
    from bentoml.grpc.utils import import_grpc
    from bentoml.grpc.utils import import_generated_stubs
    from bentoml._internal.utils import LazyLoader

    _, services = import_generated_stubs()
    pb_test, _ = import_generated_stubs(file="service_test.proto")
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")
    grpc, aio = import_grpc()


def pytest_generate_tests(metafunc: Metafunc):
    if "prometheus_interceptor" in metafunc.fixturenames:
        from bentoml._internal.configuration.containers import BentoMLContainer

        prom_dir = tempfile.mkdtemp("prometheus-multiproc-unit")
        BentoMLContainer.prometheus_multiproc_dir.set(prom_dir)
    if "prometheus_client" in metafunc.fixturenames:
        from bentoml._internal.configuration.containers import BentoMLContainer

        prom_client = BentoMLContainer.metrics_client.get()
        metafunc.parametrize("prometheus_client", [prom_client])


@pytest.fixture(scope="module")
def prometheus_interceptor():
    from bentoml.grpc.interceptors.prometheus import PrometheusServerInterceptor

    return PrometheusServerInterceptor()


@pytest.mark.asyncio
async def test_metrics_invocation(
    prometheus_interceptor: PrometheusServerInterceptor,
    mock_unary_unary_handler: MagicMock,
):
    mhandler_call_details = MagicMock(spec=grpc.HandlerCallDetails)
    mcontinuation = MagicMock(return_value=Future())
    mcontinuation.return_value.set_result(mock_unary_unary_handler)
    await prometheus_interceptor.intercept_service(mcontinuation, mhandler_call_details)
    assert mcontinuation.call_count == 1
    assert prometheus_interceptor._is_setup  # type: ignore # pylint: disable=protected-access
    assert (
        prometheus_interceptor.metrics_request_duration
        and prometheus_interceptor.metrics_request_total
        and prometheus_interceptor.metrics_request_in_progress
    )


@pytest.mark.asyncio
async def test_empty_metrics(
    prometheus_interceptor: PrometheusServerInterceptor,
    prometheus_client: PrometheusClient,
):
    # This test a branch where we change inside the handler whether or not the incoming
    # handler contains pb.Request
    # if it isn't a pb.Request, then we just pass the handler, hence metrics should be empty
    with make_standalone_server(interceptors=[prometheus_interceptor]) as (
        server,
        host_url,
    ):
        try:
            await server.start()
            async with create_channel(host_url) as channel:
                Execute = channel.unary_unary(
                    "/bentoml.testing.v1alpha1.TestService/Execute",
                    request_serializer=pb_test.ExecuteRequest.SerializeToString,
                    response_deserializer=pb_test.ExecuteResponse.FromString,
                )
                resp = t.cast(
                    t.Awaitable[pb_test.ExecuteResponse],
                    Execute(pb_test.ExecuteRequest(input="BentoML")),
                )
                await resp
                assert prometheus_client.generate_latest() == b""
        finally:
            await server.stop(None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_type,parent_set",
    [
        (
            "counter",
            ["api_name", "service_version", "http_response_code", "service_name"],
        ),
        (
            "histogram",
            ["api_name", "service_version", "http_response_code", "service_name", "le"],
        ),
        ("gauge", ["api_name", "service_version", "service_name"]),
    ],
)
async def test_metrics_interceptors(
    prometheus_interceptor: PrometheusServerInterceptor,
    prometheus_client: PrometheusClient,
    noop_service: Service,
    metric_type: str,
    parent_set: list[str],
):
    with make_standalone_server(interceptors=[prometheus_interceptor]) as (
        server,
        host_url,
    ):
        services.add_BentoServiceServicer_to_server(
            create_bento_servicer(noop_service), server
        )
        try:
            await server.start()
            async with create_channel(host_url) as channel:
                await async_client_call(
                    "noop_sync",
                    channel=channel,
                    data={"text": wrappers_pb2.StringValue(value="BentoML")},
                )
            for m in prometheus_client.text_string_to_metric_families():
                for sample in m.samples:
                    if m.type == metric_type:
                        assert set(sample.labels).issubset(set(parent_set))
                    assert (
                        "api_name" in sample.labels
                        and sample.labels["api_name"] == "noop_sync"
                    )
                    if m.type in ["counter", "histogram"]:
                        # response code is 500 because we didn't actually startup
                        # the service runner as well as running on_startup hooks.
                        # This is expected since we are testing prometheus behaviour.
                        assert sample.labels["http_response_code"] == "500"

        finally:
            await server.stop(None)
