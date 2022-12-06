from __future__ import annotations

import sys
import typing as t
import tempfile
from typing import TYPE_CHECKING
from asyncio import Future
from unittest.mock import MagicMock

import pytest

from tests.proto import service_test_pb2 as pb_test
from tests.proto import service_test_pb2_grpc as services_test
from bentoml.grpc.utils import import_grpc
from bentoml.grpc.utils import import_generated_stubs
from bentoml.testing.grpc import create_channel
from bentoml.testing.grpc import async_client_call
from bentoml.testing.grpc import make_standalone_server
from bentoml.testing.grpc import create_test_bento_servicer
from bentoml._internal.utils import LazyLoader
from tests.unit.grpc.conftest import TestServiceServicer
from bentoml.grpc.interceptors.prometheus import PrometheusServerInterceptor
from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import grpc
    from google.protobuf import wrappers_pb2

    from bentoml import Service
else:
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")
    grpc, aio = import_grpc()

prom_dir = tempfile.mkdtemp("prometheus-multiproc")
BentoMLContainer.prometheus_multiproc_dir.set(prom_dir)
interceptor = PrometheusServerInterceptor()

if "prometheus_client" in sys.modules:
    mods = [m for m in sys.modules if "prometheus_client" in m]
    list(map(lambda s: sys.modules.pop(s), mods))
    if not interceptor._is_setup:
        interceptor._setup()


@pytest.mark.asyncio
async def test_metrics_invocation(mock_unary_unary_handler: MagicMock):
    mhandler_call_details = MagicMock(spec=grpc.HandlerCallDetails)
    mcontinuation = MagicMock(return_value=Future())
    mcontinuation.return_value.set_result(mock_unary_unary_handler)
    await interceptor.intercept_service(mcontinuation, mhandler_call_details)
    assert mcontinuation.call_count == 1
    assert interceptor._is_setup  # type: ignore # pylint: disable=protected-access
    assert (
        interceptor.metrics_request_duration
        and interceptor.metrics_request_total
        and interceptor.metrics_request_in_progress
    )


@pytest.mark.asyncio
async def test_empty_metrics():
    metrics_client = BentoMLContainer.metrics_client.get()
    # This test a branch where we change inside the handler whether or not the incoming
    # handler contains pb.Request
    # if it isn't a pb.Request, then we just pass the handler, hence metrics should be empty
    with make_standalone_server(interceptors=[interceptor]) as (
        server,
        host_url,
    ):
        try:
            services_test.add_TestServiceServicer_to_server(
                TestServiceServicer(), server
            )
            await server.start()
            async with create_channel(host_url) as channel:
                Execute = channel.unary_unary(
                    "/tests.proto.TestService/Execute",
                    request_serializer=pb_test.ExecuteRequest.SerializeToString,
                    response_deserializer=pb_test.ExecuteResponse.FromString,
                )
                resp = t.cast(
                    t.Awaitable[pb_test.ExecuteResponse],
                    Execute(pb_test.ExecuteRequest(input="BentoML")),
                )
                await resp
                assert metrics_client.generate_latest() == b""
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
@pytest.mark.parametrize("protocol_version", ["v1", "v1alpha1"])
async def test_metrics_interceptors(
    simple_service: Service,
    metric_type: str,
    parent_set: list[str],
    protocol_version: str,
):
    metrics_client = BentoMLContainer.metrics_client.get()

    _, services = import_generated_stubs(protocol_version)

    with make_standalone_server(interceptors=[interceptor]) as (
        server,
        host_url,
    ):
        services.add_BentoServiceServicer_to_server(
            create_test_bento_servicer(simple_service, protocol_version), server
        )
        try:
            await server.start()
            async with create_channel(host_url) as channel:
                await async_client_call(
                    "noop_sync",
                    channel=channel,
                    data={"text": wrappers_pb2.StringValue(value="BentoML")},
                    protocol_version=protocol_version,
                )
            for m in metrics_client.text_string_to_metric_families():
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
