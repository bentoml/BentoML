"""Base class for client-side interceptor"""

import abc
from typing import Any, Callable, Iterator, NamedTuple, Optional, Sequence, Tuple, Union

import grpc


class __ClientCallDetailsFields(NamedTuple):
    method: str
    timeout: Optional[float]
    metadata: Optional[Sequence[Tuple[str, Union[str, bytes]]]]
    credentials: Optional[grpc.CallCredentials]
    wait_for_ready: Optional[bool]
    compression: Optional[grpc.Compression]


class ClientCallDetails(__ClientCallDetailsFields, grpc.ClientCallDetails):
    """refers to https://grpc.github.io/grpc/python/_modules/grpc.html#ClientCallDetails"""

    pass


class ClientInterceptorReturnType(grpc.Call, grpc.Future):
    """return type ClientInterceptor.intercept method"""

    pass


class ClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
    metaclass=abc.ABCMeta,
):
    """Base class for client-side interceptor, for custom interpretor subclass this one and override intercept"""

    @abc.abstractmethod
    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        client_call_details: grpc.ClientCallDetails,
    ) -> ClientInterceptorReturnType:
        """Override this method in custom interpretor

        This method is called for all unary and streaming RPC.
        Interceptor implementation should call `method` using a `grpc.ClientCallDetails` and `request_or_iterator` obj as params.
        `request_or_iterator` maybe type check whether its a singular request for unary RPC or an iterator for client-streaming and client-server streaming RPCs

        Args:
            method: function that proceeds with the invocation by executing the next interceptor in the chain or invoking the actual RPC handler on the channel
            request_or_iterator: RPC requse message, as protobuf message or iterator of request messages for streaming request
            client_call_details: describes the RPC to be inboked

        Returns:
            type of return should matched the return valued received by method(). This object that si both a grpc.Call and grpc.Future
            the actual results from the RPC can be invokved by calling method.results()
        """
        return method(request_or_iterator, client_call_details)

    def intercept_unary_unary(
        self,
        continuation: Callable,
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ):
        return self.intercept(_swap_args(continuation), request, client_call_details)

    def intercept_unary_stream(
        self,
        continuation: Callable,
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ):
        return self.intercept(_swap_args(continuation), request, client_call_details)

    def intercept_stream_unary(
        self,
        continuation: Callable,
        client_call_details: grpc.ClientCallDetails,
        request: Iterator[Any],
    ):
        return self.intercept(_swap_args(continuation), request, client_call_details)

    def intercept_stream_stream(
        self,
        continuation: Callable,
        client_call_details: grpc.ClientCallDetails,
        request: Iterator[Any],
    ):
        return self.intercept(_swap_args(continuation), request, client_call_details)


def _swap_args(fn: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def new_fn(x, y):
        return fn(y, x)

    return new_fn
