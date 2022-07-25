"""
Specific types for BentoService gRPC server.
"""
from __future__ import annotations

from typing import Tuple
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import Awaitable
from typing import NamedTuple
from typing import TYPE_CHECKING

import grpc


class HandlerCallDetails(
    NamedTuple("HandlerCallDetails", method=str, invocation_metadata=Tuple[str, bytes]),
    grpc.HandlerCallDetails,
):
    """Describes an RPC that has just arrived for service.

    Attributes:
    method: The method name of the RPC.
    invocation_metadata: A sequence of metadatum, a key-value pair included in the HTTP header.
                        An example is: ``('binary-metadata-bin', b'\\x00\\xFF')``
    """

    method: str
    invocation_metadata: Tuple[str, bytes]


if TYPE_CHECKING:
    from typing import Protocol

    from grpc import aio

    from bentoml.grpc.v1.service_pb2 import InferenceRequest
    from bentoml.grpc.v1.service_pb2 import InferenceResponse
    from bentoml.grpc.v1.service_pb2 import ServerLiveRequest
    from bentoml.grpc.v1.service_pb2 import ServerLiveResponse
    from bentoml.grpc.v1.service_pb2 import ServerReadyRequest
    from bentoml.grpc.v1.service_pb2 import ServerReadyResponse
    from bentoml.grpc.v1.service_pb2_grpc import BentoServiceServicer

    P = TypeVar("P", contravariant=True)

    ResponseType = InferenceResponse | ServerLiveResponse | ServerReadyResponse
    RequestType = InferenceRequest | ServerLiveRequest | ServerReadyRequest
    BentoServicerContext = aio.ServicerContext[ResponseType, RequestType]

    RequestDeserializerFn = Optional[Callable[[RequestType], object]]
    ResponseSerializerFn = Optional[Callable[[bytes], ResponseType]]

    class AsyncHandlerProtocol(Protocol[P]):
        def __call__(
            self,
            behaviour: Callable[[RequestType, BentoServicerContext], Awaitable[P]],
            request_deserializer: RequestDeserializerFn = None,
            response_serializer: ResponseSerializerFn = None,
        ) -> grpc.RpcMethodHandler:
            ...

    AsyncHandlerMethod = AsyncHandlerProtocol[ResponseType]

    class RpcMethodHandler(
        NamedTuple(
            "RpcMethodHandler",
            request_streaming=bool,
            response_streaming=bool,
            request_deserializer=RequestDeserializerFn,
            response_serializer=ResponseSerializerFn,
            unary_unary=Optional[AsyncHandlerMethod],
            unary_stream=Optional[AsyncHandlerMethod],
            stream_unary=Optional[AsyncHandlerMethod],
            stream_stream=Optional[AsyncHandlerMethod],
        ),
        grpc.RpcMethodHandler,
    ):
        """An implementation of a single RPC method."""

        request_streaming: bool
        response_streaming: bool
        request_deserializer: RequestDeserializerFn
        response_serializer: ResponseSerializerFn
        unary_unary: Optional[AsyncHandlerMethod]
        unary_stream: Optional[AsyncHandlerMethod]
        stream_unary: Optional[AsyncHandlerMethod]
        stream_stream: Optional[AsyncHandlerMethod]

    __all__ = [
        "RequestType",
        "ResponseType",
        "BentoServicerContext",
        "BentoServiceServicer",
        "HandlerCallDetails",
        "RpcMethodHandler",
    ]
