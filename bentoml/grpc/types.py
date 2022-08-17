"""
Specific types for BentoService gRPC server.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Type
    from typing import Literal
    from typing import TypeVar
    from typing import Callable
    from typing import Optional
    from typing import Annotated
    from typing import Awaitable
    from typing import NamedTuple

    import grpc
    from grpc import aio

    from bentoml.grpc.v1.service_pb2 import Request
    from bentoml.grpc.v1.service_pb2 import Response
    from bentoml.grpc.v1.service_pb2_grpc import BentoServiceServicer

    P = TypeVar("P")

    BentoServicerContext = aio.ServicerContext[Request, Response]

    RequestDeserializerFn = Callable[[Request | None], object] | None
    ResponseSerializerFn = Callable[[bytes], Response | None] | None

    HandlerMethod = Callable[[Request, BentoServicerContext], P]
    AsyncHandlerMethod = Callable[[Request, BentoServicerContext], Awaitable[P]]

    class RpcMethodHandler(
        NamedTuple(
            "RpcMethodHandler",
            request_streaming=bool,
            response_streaming=bool,
            request_deserializer=RequestDeserializerFn,
            response_serializer=ResponseSerializerFn,
            unary_unary=Optional[HandlerMethod[Response]],
            unary_stream=Optional[HandlerMethod[Response]],
            stream_unary=Optional[HandlerMethod[Response]],
            stream_stream=Optional[HandlerMethod[Response]],
        ),
        grpc.RpcMethodHandler,
    ):
        """An implementation of a single RPC method."""

        request_streaming: bool
        response_streaming: bool
        request_deserializer: RequestDeserializerFn
        response_serializer: ResponseSerializerFn
        unary_unary: Optional[HandlerMethod[Response]]
        unary_stream: Optional[HandlerMethod[Response]]
        stream_unary: Optional[HandlerMethod[Response]]
        stream_stream: Optional[HandlerMethod[Response]]

    class HandlerCallDetails(
        NamedTuple("HandlerCallDetails", method=str, invocation_metadata=aio.Metadata),
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

    ServicerImpl = TypeVar("ServicerImpl")
    Servicer = Annotated[ServicerImpl, object]
    ServicerClass = Type[Servicer[Any]]
    AddServicerFn = Callable[[Servicer[Any], aio.Server | grpc.Server], None]

    ProtoField = Literal["dataframe", "file", "json", "ndarray", "series"]

    __all__ = [
        "Request",
        "Response",
        "BentoServicerContext",
        "BentoServiceServicer",
        "HandlerCallDetails",
        "RpcMethodHandler",
    ]
