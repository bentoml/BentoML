"""
Specific types for BentoService gRPC server.
"""
from __future__ import annotations

from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Awaitable
from typing import NamedTuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol

    import grpc
    from grpc import aio

    from bentoml.grpc.v1.service_pb2 import CallRequest
    from bentoml.grpc.v1.service_pb2 import CallResponse
    from bentoml.grpc.v1.service_pb2 import ServerLiveRequest
    from bentoml.grpc.v1.service_pb2 import ServerLiveResponse
    from bentoml.grpc.v1.service_pb2 import ServerReadyRequest
    from bentoml.grpc.v1.service_pb2 import ServerReadyResponse
    from bentoml.grpc.v1.service_pb2_grpc import BentoServiceServicer

    P = TypeVar("P", contravariant=True)

    ResponseType = CallResponse | ServerLiveResponse | ServerReadyResponse
    RequestType = CallRequest | ServerLiveRequest | ServerReadyRequest
    BentoServicerContext = aio.ServicerContext[ResponseType, RequestType]

    RequestDeserializerFn = Optional[Callable[[bytes], object]]
    ResponseSerializerFn = Optional[Callable[[object], bytes]]

    class HandlerCallDetails(
        NamedTuple(
            "HandlerCallDetails", method=str, invocation_metadata=Tuple[str, bytes]
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
        invocation_metadata: Tuple[str, bytes]

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

    class ClientCallDetails(
        NamedTuple(
            "ClientCallDetails",
            method=str,
            timeout=Optional[float],
            metadata=Optional[Sequence[Tuple[str, Union[str, bytes]]]],
            credentials=Optional[grpc.CallCredentials],
            wait_for_ready=Optional[bool],
            compression=grpc.Compression,
        ),
        grpc.aio.ClientCallDetails,
    ):
        # see https://grpc.github.io/grpc/python/grpc.html#grpc.ClientCallDetails
        pass

    __all__ = [
        "RequestType",
        "ResponseType",
        "BentoServicerContext",
        "BentoServiceServicer",
        "HandlerCallDetails",
        "RpcMethodHandler",
        "ClientCallDetails",
    ]
