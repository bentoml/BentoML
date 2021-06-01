"""Interceptor a client call with prometheus"""

from timeit import default_timer

import grpc
from bentoml.yatai.metrics import (
    GRPC_SERVER_HANDLED_HISTOGRAM,
    GRPC_SERVER_HANDLED_TOTAL,
    GRPC_SERVER_STARTED_COUNTER,
    GRPC_SERVER_STREAM_MSG_RECEIVED,
    GRPC_SERVER_STREAM_MSG_SENT,
)
from bentoml.yatai.utils import (
    get_method_type,
    parse_method_name,
    wrap_interator_inc_counter,
)
from bentoml.utils.usage_stats import track


YATAI_GRPC_USAGE_EVENT_NAME = "yatai-grpc-call"


def _wrap_rpc_behaviour(handler, fn):
    """Returns a new rpc handler that wraps the given function"""
    if handler is None:
        return None

    if handler.request_streaming and handler.response_streaming:
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


class PromServerInterceptor(grpc.ServerInterceptor):  # pylint: disable=W0232
    """
    Interceptor for handling client call with metrics to prometheus
    This implementation is referred to:
    https://grpc.github.io/grpc/python/grpc.html#service-side-interceptor
    https://github.com/lchenn/py-grpc-prometheus

    Server metrics are exposed by adding the interceptor when gRPC server is started.
    Refers to
    tests/yatai/grpc_testing_service.py for a mock gRPC service
    tests/yatai/client/test_grpc_server_interceptor.py for an implementation's example

    ```python

        import grpc
        from prometheus_client import start_http_server
        from bentoml.yatai.client.interceptor import PromServerInterceptor
        ...
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=1),
            interceptors=(PromServerInterceptor(), )
            )
        start_http_server(self.prom_port)
    ```
    """

    def intercept_service(self, continuation, handler_call_details):

        # handles if call_details are None
        handler = continuation(handler_call_details)
        if handler is None:
            return handler

        # unary
        if handler.request_streaming or handler.response_streaming:
            return handler

        mn, ok = parse_method_name(handler_call_details.method)
        grpc_service_name = mn.fully_qualified_service
        grpc_method_name = mn.method

        if not ok:
            return continuation(handler_call_details)

        def metrics_wrapper(behaviour, request_streaming, response_streaming):
            def new_behaviour(request_or_iterator, servicer_context):
                grpc_type = get_method_type(request_streaming, response_streaming)

                try:
                    if request_streaming:
                        request_or_iterator = wrap_interator_inc_counter(
                            request_or_iterator,
                            GRPC_SERVER_STREAM_MSG_RECEIVED,
                            grpc_type,
                            grpc_service_name,
                            grpc_method_name,
                        )
                        request_or_iterator = wrap_interator_inc_counter(
                            request_or_iterator,
                            GRPC_SERVER_STREAM_MSG_SENT,
                            grpc_type,
                            grpc_service_name,
                            grpc_method_name,
                        )
                    else:
                        GRPC_SERVER_STARTED_COUNTER.labels(
                            grpc_type=grpc_type,
                            grpc_service=grpc_service_name,
                            grpc_method=grpc_method_name,
                        ).inc()

                    # Invoke the original RPC behaviour
                    response_or_iterator = behaviour(
                        request_or_iterator, servicer_context
                    )

                    if response_streaming:
                        sent_metrics = GRPC_SERVER_STREAM_MSG_SENT
                        response_or_iterator = wrap_interator_inc_counter(
                            response_or_iterator,
                            GRPC_SERVER_STREAM_MSG_RECEIVED,
                            grpc_type,
                            grpc_service_name,
                            grpc_method_name,
                        )
                        response_or_iterator = wrap_interator_inc_counter(
                            response_or_iterator,
                            sent_metrics,
                            grpc_type,
                            grpc_service_name,
                            grpc_method_name,
                        )
                    else:
                        GRPC_SERVER_HANDLED_TOTAL.labels(
                            grpc_type=grpc_type,
                            grpc_service=grpc_service_name,
                            grpc_method=grpc_method_name,
                            grpc_code=self._compute_status_code(servicer_context).name,
                        ).inc()

                    return response_or_iterator

                except grpc.RpcError as e:
                    GRPC_SERVER_HANDLED_TOTAL.labels(
                        grpc_type=grpc_type,
                        grpc_service=grpc_service_name,
                        grpc_method=grpc_method_name,
                        grpc_code=self._compute_error_code(e),
                    ).inc()
                    raise e

            return new_behaviour

        optional_any = _wrap_rpc_behaviour(
            continuation(handler_call_details), metrics_wrapper
        )
        return optional_any

    def _compute_status_code(self, servicer_context: grpc.ServicerContext):
        if servicer_context._state.client == "cancelled":
            return grpc.StatusCode.CANCELLED

        if servicer_context._state.code is None:
            return grpc.StatusCode.OK

        return servicer_context._state.code

    def _compute_error_code(self, grpc_exception):
        if isinstance(grpc_exception, grpc.Call):
            return grpc_exception.code().name

        return grpc.StatusCode.UNKNOWN.name


class ServiceLatencyInterceptor(grpc.ServerInterceptor):  # pylint: disable=W0232
    """Interceptor to handle service latency calls"""

    def intercept_service(self, continuation, handler_call_details):

        handler = continuation(handler_call_details)
        if handler is None:
            return handler

        # unary
        if handler.request_streaming or handler.response_streaming:
            return handler

        mn, ok = parse_method_name(handler_call_details.method)
        grpc_service_name = mn.fully_qualified_service
        grpc_method_name = mn.method

        if not ok:
            return continuation(handler_call_details)

        # pylint: disable=W0613
        def latency_wrapper(behaviour, request_streaming, response_streaming):
            def new_behaviour(request_or_iterator, servicer_context):
                start = default_timer()
                try:
                    return behaviour(request_or_iterator, servicer_context)
                finally:
                    duration = max(default_timer() - start, 0)
                    GRPC_SERVER_HANDLED_HISTOGRAM.labels(
                        grpc_type='UNARY',
                        grpc_service=grpc_service_name,
                        grpc_method=grpc_method_name,
                    ).observe(duration)
                    track(
                        YATAI_GRPC_USAGE_EVENT_NAME,
                        {"method": grpc_method_name, "duration": duration},
                    )

            return new_behaviour

        return _wrap_rpc_behaviour(continuation(handler_call_details), latency_wrapper)
