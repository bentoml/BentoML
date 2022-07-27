"""
Specific types for BentoService gRPC server.
"""
from __future__ import annotations

from typing import Any
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import NamedTuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol

    import grpc
    from grpc import aio

    from bentoml.grpc.v1.service_pb2 import Request
    from bentoml.grpc.v1.service_pb2 import Response
    from bentoml.grpc.v1.service_pb2_grpc import BentoServiceServicer

    P_con = TypeVar("P_con", contravariant=True)

    BentoServicerContext = aio.ServicerContext[Response, Request]

    RequestDeserializerFn = Callable[[Request | None], object] | None
    ResponseSerializerFn = Callable[[bytes], Response | None] | None

    HandlerMethod = Callable[[Request, BentoServicerContext], P_con]

    class HandlerFactoryProtocol(Protocol[P_con]):
        def __call__(
            self,
            behaviour: HandlerMethod[P_con],
            request_deserializer: RequestDeserializerFn = None,
            response_serializer: ResponseSerializerFn = None,
        ) -> grpc.RpcMethodHandler:
            ...

    HandlerFactoryFn = HandlerFactoryProtocol[Any]

    class RpcMethodHandler(
        NamedTuple(
            "RpcMethodHandler",
            request_streaming=bool,
            response_streaming=bool,
            request_deserializer=RequestDeserializerFn,
            response_serializer=ResponseSerializerFn,
            unary_unary=Optional[aio.UnaryUnaryMultiCallable],
            unary_stream=Optional[aio.UnaryStreamMultiCallable],
            stream_unary=Optional[aio.StreamUnaryMultiCallable],
            stream_stream=Optional[aio.StreamStreamMultiCallable],
        ),
        grpc.RpcMethodHandler,
    ):
        """An implementation of a single RPC method."""

        request_streaming: bool
        response_streaming: bool
        request_deserializer: RequestDeserializerFn
        response_serializer: ResponseSerializerFn
        unary_unary: Optional[aio.UnaryUnaryMultiCallable]
        unary_stream: Optional[aio.UnaryStreamMultiCallable]
        stream_unary: Optional[aio.StreamUnaryMultiCallable]
        stream_stream: Optional[aio.StreamStreamMultiCallable]

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

    __all__ = [
        "Request",
        "Response",
        "BentoServicerContext",
        "BentoServiceServicer",
        "HandlerCallDetails",
        "RpcMethodHandler",
    ]
