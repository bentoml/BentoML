"""Base class for server-side interceptor"""
import abc
from typing import Any, Callable

import grpc
from bentoml.yatai.client.interceptor.utils import get_factory_and_method


class ServerInterceptor(grpc.ServerInterceptor, metaclass=abc.ABCMeta):
    """Base class for server-side implementation for interceptor"""

    @abc.abstractmethod
    def intercept(
        self,
        method: Callable,
        request: Any,
        servicer_context: grpc.ServicerContext,
        method_name: str,
    ):
        """same with client, implementation should override this function

        One should call method(request,servicer_context) to invoke the next handler (either and rpc_handler or next interceptor)


        Args:
            method: either grpc.RpcMethodHandler or the next interceptor in the chain
            request: RPC request, as in protobuf message, or iterator of multiple RPC requests
            servicer_context: ServicerContext pass by gRPC to the service
            method_name: string in a form "/protobuf.package.Service/Method"
        Returns:
            method(request, context) which aligns with rpc method response, as a protobuf message
        """

        return method(request, servicer_context)

    def intercept_service(self, continuation, handler_call_details):
        """Shouldn't override this method. Implementation from grpc.ServerInterceptor"""
        next_handler = continuation(handler_call_details)

        handler_factory, next_handler_method = get_factory_and_method(next_handler)

        def invoke_intercept_method(request, servicer_context):
            method_name = handler_call_details.method
            return self.intercept(
                next_handler_method, request, servicer_context, method_name
            )

        return handler_factory(
            invoke_intercept_method,
            request_deserializer=next_handler.request_deserializer,
            response_serializer=next_handler.response_serializer,
        )
