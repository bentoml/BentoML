from __future__ import annotations

import typing as t

import pytest
from google.protobuf import wrappers_pb2

import bentoml
from _bentoml_impl.server.grpc.servicer.v1 import create_bento_servicer
from bentoml.exceptions import BentoMLException
from bentoml.grpc.utils import import_generated_stubs
from bentoml.grpc.utils import import_grpc

pb, _ = import_generated_stubs("v1")
grpc, aio = import_grpc()


@bentoml.service
class Greeter:
    @bentoml.api
    def greet(self, name: str) -> str:
        return f"hello {name}"

    @bentoml.api
    async def agreet(self, name: str) -> str:
        return f"hello {name}"

    @bentoml.api
    async def stream_greet(self, name: str) -> t.AsyncGenerator[str, None]:
        yield f"hello {name}"

    @bentoml.api(batchable=True)
    def batch_greet(self, name: list[str]) -> list[str]:
        return [f"hello {n}" for n in name]

    @bentoml.task
    def long_job(self, name: str) -> str:
        return name


class FakeContext:
    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    async def abort(self, code: grpc.StatusCode, details: str = "") -> t.NoReturn:
        self.code = code
        self.details = details
        raise aio.AbortError()


@pytest.fixture
def servicer():
    return create_bento_servicer(Greeter)


@pytest.mark.asyncio
async def test_call_greet(servicer):
    ctx = FakeContext()
    request = pb.Request(
        api_name="greet", text=wrappers_pb2.StringValue(value="world")
    )
    response = await servicer.Call(request, ctx)
    assert response is not None
    assert response.text.value == "hello world"


@pytest.mark.asyncio
async def test_call_async_greet(servicer):
    ctx = FakeContext()
    request = pb.Request(
        api_name="agreet", text=wrappers_pb2.StringValue(value="world")
    )
    response = await servicer.Call(request, ctx)
    assert response is not None
    assert response.text.value == "hello world"


@pytest.mark.asyncio
async def test_unknown_api_name_aborts(servicer):
    ctx = FakeContext()
    request = pb.Request(
        api_name="missing", text=wrappers_pb2.StringValue(value="x")
    )
    with pytest.raises(aio.AbortError):
        await servicer.Call(request, ctx)
    assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
    assert ctx.details is not None
    assert "api_name" in ctx.details


@pytest.mark.asyncio
async def test_streaming_api_unimplemented(servicer):
    ctx = FakeContext()
    request = pb.Request(
        api_name="stream_greet", text=wrappers_pb2.StringValue(value="world")
    )
    with pytest.raises(aio.AbortError):
        await servicer.Call(request, ctx)
    assert ctx.code == grpc.StatusCode.UNIMPLEMENTED
    assert ctx.details is not None
    assert "streaming" in ctx.details


@pytest.mark.asyncio
async def test_batchable_api_unimplemented(servicer):
    ctx = FakeContext()
    request = pb.Request(api_name="batch_greet")
    with pytest.raises(aio.AbortError):
        await servicer.Call(request, ctx)
    assert ctx.code == grpc.StatusCode.UNIMPLEMENTED
    assert ctx.details is not None
    assert "batchable" in ctx.details


@pytest.mark.asyncio
async def test_task_api_unimplemented(servicer):
    ctx = FakeContext()
    request = pb.Request(
        api_name="long_job", text=wrappers_pb2.StringValue(value="world")
    )
    with pytest.raises(aio.AbortError):
        await servicer.Call(request, ctx)
    assert ctx.code == grpc.StatusCode.UNIMPLEMENTED
    assert ctx.details is not None
    assert "task" in ctx.details


@pytest.mark.asyncio
async def test_service_metadata(servicer):
    ctx = FakeContext()
    meta = await servicer.ServiceMetadata(pb.ServiceMetadataRequest(), ctx)
    assert meta.name == "Greeter"
    names = {api.name for api in meta.apis}
    assert {"greet", "agreet", "stream_greet", "batch_greet", "long_job"} <= names
    greet = next(api for api in meta.apis if api.name == "greet")
    assert greet.input.descriptor_id == "bentoml.sdk.IODescriptor"
    assert greet.output.descriptor_id == "bentoml.sdk.IODescriptor"


def test_serve_grpc_rejects_v1alpha1():
    from _bentoml_impl.server.serving import serve_grpc

    with pytest.raises(BentoMLException, match="protocol v1"):
        serve_grpc("service.py:Greeter", protocol_version="v1alpha1")
