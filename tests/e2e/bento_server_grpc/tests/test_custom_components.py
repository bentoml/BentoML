from __future__ import annotations

import pytest
from grpc import aio
from grpc_health.v1 import health_pb2 as pb_health
from google.protobuf import wrappers_pb2

import bentoml


def test_success_invocation_custom_servicer(host: str) -> None:
    client = bentoml.client.GrpcClient.from_url(host)
    health = client.health("bentoml.grpc.v1.BentoService")
    assert health.status == pb_health.HealthCheckResponse.SERVING  # type: ignore ( no generated enum types)


@pytest.mark.asyncio
async def test_trailing_metadata_interceptors(host: str) -> None:
    client = bentoml.client.GrpcClient.from_url(host)
    async with client.create_channel() as channel:
        stubs = client._services.BentoServiceStub(channel)
        resp = stubs.Call(
            client._pb.Request(
                api_name="bonjour", text=wrappers_pb2.StringValue(value="BentoML")
            )
        )
        trailing_metadata = await resp.trailing_metadata()
        assert trailing_metadata == aio.Metadata.from_tuple(
            (("usage", "NLP"), ("accuracy_score", "0.8247"))
        )
