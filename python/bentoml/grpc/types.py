# pragma: no cover
"""
Specific types for BentoService gRPC server.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typing as t
    from functools import partial

    import grpc
    from grpc import aio

    from bentoml.grpc.v1.service_pb2 import Request
    from bentoml.grpc.v1.service_pb2 import Response
    from bentoml.grpc.v1.service_pb2_grpc import BentoServiceServicer

    P = t.TypeVar("P")

    BentoServicerContext = aio.ServicerContext[Request, Response]

    RequestDeserializerFn = t.Callable[[Request | None], object] | None
    ResponseSerializerFn = t.Callable[[bytes], Response | None] | None

    HandlerMethod = t.Callable[[Request, BentoServicerContext], P]
    AsyncHandlerMethod = t.Callable[[Request, BentoServicerContext], t.Awaitable[P]]

    class RpcMethodHandler(
        t.NamedTuple(
            "RpcMethodHandler",
            request_streaming=bool,
            response_streaming=bool,
            request_deserializer=RequestDeserializerFn,
            response_serializer=ResponseSerializerFn,
            unary_unary=t.Optional[HandlerMethod[Response]],
            unary_stream=t.Optional[HandlerMethod[Response]],
            stream_unary=t.Optional[HandlerMethod[Response]],
            stream_stream=t.Optional[HandlerMethod[Response]],
        ),
        grpc.RpcMethodHandler,
    ):
        """An implementation of a single RPC method."""

        request_streaming: bool
        response_streaming: bool
        request_deserializer: RequestDeserializerFn
        response_serializer: ResponseSerializerFn
        unary_unary: t.Optional[HandlerMethod[Response]]
        unary_stream: t.Optional[HandlerMethod[Response]]
        stream_unary: t.Optional[HandlerMethod[Response]]
        stream_stream: t.Optional[HandlerMethod[Response]]

    class HandlerCallDetails(
        t.NamedTuple(
            "HandlerCallDetails", method=str, invocation_metadata=aio.Metadata
        ),
        grpc.HandlerCallDetails,
    ):
        """Describes an RPC that has just arrived for service.

        Attributes:
        method: The method name of the RPC.
        invocation_metadata: A sequence of metadatum, a key-value pair included in the HTTP header.
                            An example is: ``('binary-metadata-bin', b'\\x00\\xFF')``
        """

        method: str
        invocation_metadata: aio.Metadata

    # Servicer types
    ServicerImpl = t.TypeVar("ServicerImpl")
    Servicer = t.Annotated[ServicerImpl, object]
    ServicerClass = t.Type[Servicer[t.Any]]
    AddServicerFn = t.Callable[[Servicer[t.Any], aio.Server | grpc.Server], None]

    # accepted proto fields
    ProtoField = t.Annotated[
        str,
        t.Literal[
            "dataframe",
            "file",
            "json",
            "ndarray",
            "series",
            "text",
            "multipart",
            "serialized_bytes",
        ],
    ]

    Interceptors = list[
        t.Callable[[], aio.ServerInterceptor] | partial[aio.ServerInterceptor]
    ]

    __all__ = [
        "Request",
        "Response",
        "BentoServicerContext",
        "BentoServiceServicer",
        "HandlerCallDetails",
        "RpcMethodHandler",
    ]
