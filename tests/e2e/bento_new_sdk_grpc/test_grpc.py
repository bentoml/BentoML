from __future__ import annotations

import asyncio
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import psutil
import pytest
from google.protobuf import wrappers_pb2

import bentoml
from bentoml.grpc.utils import import_generated_stubs
from bentoml.grpc.utils import import_grpc
from bentoml.testing.grpc import async_client_call
from bentoml.testing.grpc import create_channel
from bentoml.testing.grpc import make_pb_ndarray

pytest.importorskip("grpc")

grpc, aio = import_grpc()
pb, _ = import_generated_stubs("v1")

PROJECT_DIR = Path(__file__).parent
PORT = 38765


pytestmark = pytest.mark.skipif(
    psutil.WINDOWS, reason="gRPC is not supported on Windows."
)


def _host_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.hostname}:{parsed.port}"


async def _wait_until_ready(host_url: str, timeout: float = 100) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            async with aio.insecure_channel(host_url) as channel:
                await asyncio.wait_for(channel.channel_ready(), timeout=2)
                return
        except Exception as exc:
            last_err = exc
            await asyncio.sleep(0.5)
    raise TimeoutError(f"gRPC server at {host_url} was not ready") from last_err


@pytest.mark.asyncio
async def test_unary_greet_and_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENTOML_CONFIG_OPTIONS", "services.metrics.enabled=false")
    with bentoml.serve(
        "service.py:MyService",
        working_dir=str(PROJECT_DIR),
        port=PORT,
        server_type="grpc",
        production=False,
        args={"greeting": "hello"},
    ) as server:
        host_url = _host_url(server.url)
        await _wait_until_ready(host_url, timeout=100)
        async with create_channel(host_url) as channel:
            await async_client_call(
                "greet",
                channel=channel,
                data={"text": wrappers_pb2.StringValue(value="world")},
                assert_code=grpc.StatusCode.OK,
                assert_data=lambda resp: resp.text.value == "hello world",
            )
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            await async_client_call(
                "predict",
                channel=channel,
                data={"ndarray": make_pb_ndarray(arr)},
                assert_code=grpc.StatusCode.OK,
                assert_data=lambda resp: np.allclose(
                    resp.ndarray.float_values, [2.0, 4.0, 6.0]
                ),
            )
            call_rpc = channel.unary_unary(
                "/bentoml.grpc.v1.BentoService/Call",
                request_serializer=pb.Request.SerializeToString,
                response_deserializer=pb.Response.FromString,
            )
            context_call = call_rpc(
                pb.Request(
                    api_name="context_greet",
                    text=wrappers_pb2.StringValue(value="world"),
                ),
                metadata=(("x-request-source", "grpc-client"),),
            )
            context_response = await context_call
            assert context_response.text.value == "grpc-client world"
            assert dict(await context_call.trailing_metadata())[
                "x-response-source"
            ] == "bentoml-context"
            await async_client_call(
                "missing",
                channel=channel,
                data={"text": wrappers_pb2.StringValue(value="x")},
                assert_code=grpc.StatusCode.INVALID_ARGUMENT,
            )
            await async_client_call(
                "stream_greet",
                channel=channel,
                data={"text": wrappers_pb2.StringValue(value="world")},
                assert_code=grpc.StatusCode.UNIMPLEMENTED,
            )
