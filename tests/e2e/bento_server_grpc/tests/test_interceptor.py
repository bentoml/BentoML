from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import grpc
import pytest
from opentelemetry import trace as trace_api
from google.protobuf import wrappers_pb2
from opentelemetry.semconv.trace import SpanAttributes

from bentoml.testing.grpc import create_channel
from bentoml.grpc.v1alpha1 import service_pb2 as pb
from bentoml.grpc.v1alpha1 import service_pb2_grpc as services
from bentoml.grpc.v1alpha1 import service_test_pb2 as pb_test
from bentoml.grpc.v1alpha1 import service_test_pb2_grpc as services_test
from bentoml.testing.utils import async_request
from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import opentelemetry.instrumentation.grpc as otel_grpc
    from opentelemetry.sdk.trace import Span
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
else:
    otel_grpc = LazyLoader("otel_grpc", globals(), "opentelemetry.instrumentation.grpc")


@pytest.mark.asyncio
async def test_prometheus_interceptor(host: str) -> None:
    from bentoml._internal.configuration.containers import BentoMLContainer

    metrics_client = BentoMLContainer.metrics_client.get()

    # Currently, tests doesn't change default metrics ports, which is 3001
    async with create_channel(host) as channel:
        client = services.BentoServiceStub(channel)  # type: ignore (no async types)
        await client.Call(
            pb.Request(
                api_name="bonjour", text=wrappers_pb2.StringValue(value="BentoML")
            )
        )
        await async_request(
            "GET",
            "http://0.0.0.0:3001",
            assert_status=200,
            assert_data=lambda d: d in metrics_client.generate_latest(),
        )


def assert_span_has_attributes(span: Span, attributes: dict[str, t.Any]):
    assert span.attributes
    for key, value in attributes.items():
        assert key in span.attributes
        assert span.attributes[key] == value


def sanity_check(span: Span, name: str):
    # check if span name is the same as rpc_call
    assert span.name == name
    assert span.kind == trace_api.SpanKind.SERVER

    # sanity check
    assert span.instrumentation_info.name == otel_grpc.__name__
    assert span.instrumentation_info.version == otel_grpc.__version__


@pytest.mark.skip("Currently broken test. (Revisit after experimental release)")
@pytest.mark.otel
@pytest.mark.asyncio
async def test_otel_interceptor(memory_exporter: InMemorySpanExporter, host: str):
    async with create_channel(host) as channel:
        stub = services_test.TestServiceStub(channel)  # type: ignore (no async types)
        await stub.Execute(pb_test.ExecuteRequest(input="BentoML"))

        spans_list = t.cast("list[Span]", memory_exporter.get_finished_spans())
        assert len(spans_list) == 1
        # We only care about the second span, which is the rpc_call
        span = spans_list[0]
        service_name = "bentoml.testing.v1alpha1.TestService"

        sanity_check(span, f"/{service_name}/Execute")
        assert_span_has_attributes(
            span,
            {
                SpanAttributes.NET_PEER_IP: "[::1]",
                SpanAttributes.NET_PEER_NAME: "localhost",
                SpanAttributes.RPC_METHOD: "Execute",
                SpanAttributes.RPC_SERVICE: service_name,
                SpanAttributes.RPC_SYSTEM: "grpc",
                SpanAttributes.RPC_GRPC_STATUS_CODE: grpc.StatusCode.OK.value[0],
            },
        )
