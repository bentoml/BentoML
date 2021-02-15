"""Interceptor a client call with prometheus"""

from timeit import default_timer

import grpc
from bentoml.yatai.client.interceptor.base_server_interceptor import ServerInterceptor
from bentoml.yatai.client.interceptor.utils import (
    get_factory_and_method,
    get_method_type,
    parse_method_name,
    wrap_interator_inc_counter,
)
from bentoml.yatai.metrics_exporter.server_metrics import (
    GRPC_SERVER_HANDLED_HISTOGRAM,
    GRPC_SERVER_HANDLED_TOTAL,
    GRPC_SERVER_STARTED_COUNTER,
    GRPC_SERVER_STREAM_MSG_RECEIVED,
    GRPC_SERVER_STREAM_MSG_SENT,
)


# request are either RPC request as protobuf or iterator of RPC request
class PromServerInterceptor(ServerInterceptor):
    """Interceptor for handling client call with metrics to prometheus"""

    def __init__(self, enable_handling_time_historgram=False):
        self.enable_handling_time_historgram = enable_handling_time_historgram

    # this will be metrics_wrapper
    def intercept(self, method, request, servicer_context, method_name):
        start = default_timer()
        # method should be type grpc.RpcMethodHandler
        _, grpc_service_name, grpc_method_name = parse_method_name(method_name)
        handler, _ = get_factory_and_method(method)
        grpc_type = get_method_type(handler)

        try:
            if handler.request_streaming:
                request = wrap_interator_inc_counter(
                    request,
                    GRPC_SERVER_STREAM_MSG_RECEIVED,
                    grpc_type,
                    grpc_service_name,
                    grpc_method_name,
                )
                request = wrap_interator_inc_counter(
                    request,
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
            response_or_iterator = method(request, servicer_context)

            if handler.request_streaming:
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
                self.increase_grpc_server_handled_total_counter(
                    grpc_type,
                    grpc_service_name,
                    grpc_method_name,
                    self._compute_status_code(servicer_context).name,
                )

            return response_or_iterator

        except grpc.RpcError as e:
            self.increase_grpc_server_handled_total_counter(
                grpc_type,
                grpc_service_name,
                grpc_method_name,
                self._compute_error_code(e).name,
            )
            raise e

        finally:
            if not handler.response_streaming:
                if self.enable_handling_time_historgram:
                    GRPC_SERVER_HANDLED_HISTOGRAM.labels(
                        grpc_type=grpc_type,
                        grpc_service=grpc_service_name,
                        grpc_method=grpc_method_name,
                    ).observe(max(default_timer() - start, 0))

    def _compute_status_code(self, servicer_context: grpc.ServicerContext):
        if servicer_context._state_client == "cancelled":
            return grpc.StatusCode.CANCELLED
        if servicer_context._state.code is None:
            return grpc.StatusCode.OK
        return servicer_context._state.code

    def _compute_error_code(self, grpc_exception):
        if isinstance(grpc_exception, grpc.Call):
            return grpc_exception.code().name
        return grpc.StatusCode.UNKNOWN.name

    def increase_grpc_server_handled_total_counter(
        self, grpc_type, grpc_service_name, grpc_method_name, grpc_code
    ):
        GRPC_SERVER_HANDLED_TOTAL.labels(
            grpc_type=grpc_type,
            grpc_service=grpc_service_name,
            grpc_method=grpc_method_name,
            grpc_code=grpc_code,
        ).inc()
