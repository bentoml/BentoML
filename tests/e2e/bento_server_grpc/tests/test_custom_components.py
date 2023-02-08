from __future__ import annotations

import typing as t

import pytest
from grpc import aio
from grpc_health.v1 import health_pb2 as pb_health
from google.protobuf import wrappers_pb2

from bentoml.grpc.utils import import_generated_stubs
from bentoml.testing.grpc import create_channel
from bentoml.testing.grpc import async_client_call


@pytest.mark.asyncio
async def test_success_invocation_custom_servicer(host: str) -> None:
    async with create_channel(host) as channel:
        HealthCheck = channel.unary_unary(
            "/grpc.health.v1.Health/Check",
            request_serializer=pb_health.HealthCheckRequest.SerializeToString,  # type: ignore (no grpc_health type)
            response_deserializer=pb_health.HealthCheckResponse.FromString,  # type: ignore (no grpc_health type)
        )
        health = await t.cast(
            t.Awaitable[pb_health.HealthCheckResponse],
            HealthCheck(
                pb_health.HealthCheckRequest(service="bentoml.grpc.v1.BentoService")
            ),
        )
        assert health.status == pb_health.HealthCheckResponse.SERVING  # type: ignore ( no generated enum types)


@pytest.mark.asyncio
async def test_trailing_metadata_interceptors(host: str) -> None:
    async with create_channel(host) as channel:
        await async_client_call(
            "bonjour",
            channel=channel,
            data={"text": wrappers_pb2.StringValue(value="BentoML")},
            assert_trailing_metadata=aio.Metadata.from_tuple(
                (("usage", "NLP"), ("accuracy_score", "0.8247"))
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ["v1", "v1alpha1"])
async def test_grpc_context(host: str, protocol_version: str) -> None:
    if t.TYPE_CHECKING:
        from bentoml.grpc.v1 import service_pb2 as pb
    else:
        pb, _ = import_generated_stubs(protocol_version)

    async with create_channel(host) as channel:
        Call = channel.unary_unary(
            f"/bentoml.grpc.{protocol_version}.BentoService/Call",
            request_serializer=pb.Request.SerializeToString,
            response_deserializer=pb.Response.FromString,
        )
        output: aio.UnaryUnaryCall[pb.Request, pb.Response] = Call(
            pb.Request(
                api_name="echo_check_grpc_context",
                text=wrappers_pb2.StringValue(value="BentoML"),
            )
        )
        initial_metadata = await output.initial_metadata()
        assert initial_metadata == aio.Metadata.from_tuple((("foo", "bar"),))
