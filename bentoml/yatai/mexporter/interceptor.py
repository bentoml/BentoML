import timeit

import grpc
from server_metrics import (
    GRPC_SERVER_HANDLED_LATENCY_SECONDS,
    GRPC_SERVER_HANDLED_TOTAL_COUNTER,
    GRPC_SERVER_MSG_RECEIVED_TOTAL_COUNTER,
    GRPC_SERVER_MSG_SENT_TOTAL_COUNTER,
    GRPC_SERVER_STARTED_TOTAL_COUNTER,
)

UNARY = "UNARY"
SERVER_STREAMING = "SERVER_STREAMING"
CLIENT_STREAMING = "CLIENT_STREAMING"
BIDI_STREAMING = "BIDI_STREAMING"
UNKNOWN = "UNKNOWN"


def wrap_iterator_inc_counter(
    iterator, counter, grpc_type, grpc_service_name, grpc_method_name
):
    """Wraps an iterator and collect metrics."""

    for item in iterator:
        counter.labels(
            grpc_type=grpc_type,
            grpc_service=grpc_service_name,
            grpc_method=grpc_method_name,
        ).inc()
        yield item


def get_method_type(request_streaming, response_streaming):
    """
  Infers the method type from if the request or the response is streaming.
  # The Method type is coming from:
  # https://grpc.io/grpc-java/javadoc/io/grpc/MethodDescriptor.MethodType.html
  """
    if request_streaming and response_streaming:
        return BIDI_STREAMING
    elif request_streaming and not response_streaming:
        return CLIENT_STREAMING
    elif not request_streaming and response_streaming:
        return SERVER_STREAMING
    return UNARY


def split_method_call(handler_call_details):
    """
  Infers the grpc service and method name from the handler_call_details.
  """

    # e.g. /package.ServiceName/MethodName
    parts = handler_call_details.method.split("/")
    if len(parts) < 3:
        return "", "", False

    grpc_service_name, grpc_method_name = parts[1:3]
    return grpc_service_name, grpc_method_name, True


class Interceptor(grpc.ServerInterceptor):
    def intercept_service(self, cont, handler_call_details):
        """Intecepts the server function calls."""
        grpc_service_name, grpc_method_name, _ = split_method_call(handler_call_details)

        def metrics_wrapper(behavior, request_streaming, response_streaming):
            def new_behaviour(request_or_iterator, service_context):
                start = timeit.default_timer()
                grpc_type = get_method_type(request_streaming, response_streaming)
                try:
                    if request_streaming:
                        request_or_iterator = wrap_iterator_inc_counter(
                            request_or_iterator,
                            GRPC_SERVER_MSG_RECEIVED_TOTAL_COUNTER,
                            grpc_type,
                            grpc_service_name,
                            grpc_method_name,
                        )
                    else:
                        GRPC_SERVER_STARTED_TOTAL_COUNTER.labels(
                            grpc_type=grpc_type,
                            grpc_service=grpc_service_name,
                            grpc_method=grpc_method_name,
                        ).inc()

                    response_or_iterator = behavior(
                        request_or_iterator, service_context
                    )

                    if response_streaming:
                        response_or_iterator = wrap_iterator_inc_counter(
                            response_or_iterator,
                            GRPC_SERVER_MSG_SENT_TOTAL_COUNTER,
                            grpc_type,
                            grpc_service_name,
                            grpc_method_name,
                        )
                    else:
                        GRPC_SERVER_HANDLED_TOTAL_COUNTER.labels(
                            grpc_type=grpc_type,
                            grpc_service_name=grpc_service_name,
                            grpc_method=grpc_method_name,
                            code=self.compute_status_code(service_context).name,
                        ).inc()
                    return response_or_iterator
                except grpc.RpcError as e:
                    GRPC_SERVER_HANDLED_TOTAL_COUNTER.labels(
                        grpc_type=grpc_type,
                        grpc_service=grpc_service_name,
                        grpc_method=grpc_method_name,
                        code=self.compute_error_code(e),
                    ).inc()
                    raise e
                finally:
                    if not response_streaming:
                        GRPC_SERVER_HANDLED_LATENCY_SECONDS.labels(
                            grpc_type=grpc_type,
                            grpc_service=grpc_service_name,
                            grpc_method=grpc_method_name,
                        ).observe(max(timeit.default_timer() - start, 0))

            return new_behaviour

        return self.wrap_rpc_behaviour(cont(handler_call_details), metrics_wrapper)

    def compute_status_code(self, service_context):
        if service_context._state.client == "cancelled":
            return grpc.StatusCode.CANCELLED

        if service_context._state.code is None:
            return grpc.StatusCode.OK

        return service_context._state.code

    def compute_error_code(self, grpc_exception):
        if isinstance(grpc_exception, grpc.Call):
            return grpc_exception.code().name

        return grpc.StatusCode.UNKNOWN.name

    def wrap_rpc_behaviour(self, handler, fn):
        """returns new rpc handler that wraps the given function"""

        if handler is None:
            return None

        if handler.request_streaming or handler.response_streaming:
            behavior_fn = handler.stream_stream
            handler_factory = grpc.stream_stream_rpc_method_handler
        elif handler.request_streaming and not handler.response_streaming:
            behavior_fn = handler.stream_unary
            handler_factory = grpc.stream_unary_rpc_method_handler
        elif not handler.request_streaming and handler.response_streaming:
            behavior_fn = handler.unary_stream
            handler_factory = grpc.unary_stream_rpc_method_handler
        else:
            behavior_fn = handler.unary_unary
            handler_factory = grpc.unary_unary_rpc_method_handler

        return handler_factory(
            fn(behavior_fn, handler.request_streaming, handler.response_streaming),
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )
