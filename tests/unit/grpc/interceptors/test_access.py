# pylint: disable=unused-argument
from __future__ import annotations

import typing as t
import logging
import functools
from typing import TYPE_CHECKING

import pytest

from bentoml.grpc.utils import import_grpc
from bentoml.grpc.utils import wrap_rpc_handler
from bentoml.grpc.utils import import_generated_stubs
from bentoml.testing.grpc import create_channel
from bentoml.testing.grpc import async_client_call
from bentoml.testing.grpc import create_bento_servicer
from bentoml.testing.grpc import make_standalone_server
from bentoml._internal.utils import LazyLoader
from bentoml.grpc.interceptors.access import AccessLogServerInterceptor
from bentoml.grpc.interceptors.opentelemetry import AsyncOpenTelemetryServerInterceptor

if TYPE_CHECKING:
    import grpc
    from grpc import aio
    from _pytest.logging import LogCaptureFixture
    from google.protobuf import wrappers_pb2

    from bentoml import Service
    from bentoml.grpc.types import Request
    from bentoml.grpc.types import Response
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import AsyncHandlerMethod
    from bentoml.grpc.types import HandlerCallDetails
    from bentoml.grpc.types import BentoServicerContext
    from bentoml.grpc.v1alpha1 import service_pb2_grpc as services
    from bentoml.grpc.v1alpha1 import service_test_pb2 as pb_test
    from bentoml.grpc.v1alpha1 import service_test_pb2_grpc as services_test
else:
    _, services = import_generated_stubs()
    pb_test, services_test = import_generated_stubs(file="service_test.proto")
    grpc, aio = import_grpc()
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")


class AppendMetadataInterceptor(aio.ServerInterceptor):
    def __init__(self, *metadata: tuple[str, t.Any]):
        self._metadata = tuple(metadata)

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        handler = await continuation(handler_call_details)
        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        def wrapper(behaviour: AsyncHandlerMethod[Response]):
            @functools.wraps(behaviour)
            async def new_behaviour(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:
                context.set_trailing_metadata(aio.Metadata.from_tuple(self._metadata))
                return await behaviour(request, context)

            return new_behaviour

        return wrap_rpc_handler(wrapper, handler)


@pytest.mark.asyncio
@pytest.mark.usefixtures("propagate_logs")
async def test_success_logs(caplog: LogCaptureFixture):
    with make_standalone_server(
        # we need to also setup opentelemetry interceptor
        # to make sure the access log is correctly setup.
        interceptors=[
            AsyncOpenTelemetryServerInterceptor(),
            AccessLogServerInterceptor(),
        ]
    ) as (server, host_url):
        try:
            await server.start()
            with caplog.at_level(logging.INFO, "bentoml.access"):
                async with create_channel(host_url) as channel:
                    stub = services_test.TestServiceStub(channel)
                    await stub.Execute(pb_test.ExecuteRequest(input="BentoML"))
            assert (
                "(scheme=http,path=/bentoml.testing.v1alpha1.TestService/Execute,type=application/grpc,size=9) (http_status=200,grpc_status=0,type=application/grpc,size=17)"
                in caplog.text
            )

        finally:
            await server.stop(None)


@pytest.mark.asyncio
@pytest.mark.usefixtures("propagate_logs")
async def test_trailing_metadata(caplog: LogCaptureFixture):
    with make_standalone_server(
        # we need to also setup opentelemetry interceptor
        # to make sure the access log is correctly setup.
        interceptors=[
            AsyncOpenTelemetryServerInterceptor(),
            AppendMetadataInterceptor(("content-type", "application/grpc+python")),
            AccessLogServerInterceptor(),
        ]
    ) as (server, host_url):
        try:
            await server.start()
            with caplog.at_level(logging.INFO, "bentoml.access"):
                async with create_channel(host_url) as channel:
                    stub = services_test.TestServiceStub(channel)
                    await stub.Execute(pb_test.ExecuteRequest(input="BentoML"))
            assert "type=application/grpc+python" in caplog.text
        finally:
            await server.stop(None)


@pytest.mark.asyncio
@pytest.mark.usefixtures("propagate_logs")
async def test_access_log_exception(caplog: LogCaptureFixture, simple_service: Service):
    with make_standalone_server(
        # we need to also setup opentelemetry interceptor
        # to make sure the access log is correctly setup.
        interceptors=[
            AsyncOpenTelemetryServerInterceptor(),
            AccessLogServerInterceptor(),
        ]
    ) as (server, host_url):
        services.add_BentoServiceServicer_to_server(
            create_bento_servicer(simple_service), server
        )
        try:
            await server.start()
            with caplog.at_level(logging.INFO):
                async with create_channel(host_url) as channel:
                    await async_client_call(
                        "invalid",
                        channel=channel,
                        data={"text": wrappers_pb2.StringValue(value="asdf")},
                        assert_code=grpc.StatusCode.INTERNAL,
                    )
            assert (
                "(scheme=http,path=/bentoml.grpc.v1alpha1.BentoService/Call,type=application/grpc,size=17) (http_status=500,grpc_status=13,type=application/grpc,size=0)"
                in caplog.text
            )
        finally:
            await server.stop(None)
