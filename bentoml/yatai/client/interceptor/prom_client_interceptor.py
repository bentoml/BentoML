from timeit import default_timer
from typing import Any, Callable

from bentoml.yatai.client.interceptor.base_client_interceptor import (
    ClientCallDetails,
    ClientInterceptor,
)
from bentoml.yatai.client.interceptor.utils import (
    get_factory_and_method,
    get_method_type,
    parse_method_name,
    wrap_interator_inc_counter,
)
from bentoml.yatai.metrics_exporter.client_metrics import (
    GRPC_CLIENT_HANDLED_COUNTER,
    GRPC_CLIENT_HANDLED_HISTORGRAM,
    GRPC_CLIENT_STARTED_COUNTER,
    GRPC_CLIENT_STREAM_MSG_RECEIVED,
    GRPC_CLIENT_STREAM_MSG_SENT,
    GRPC_CLIENT_STREAM_RECV_HISTOGRAM,
    GRPC_CLIENT_STREAM_SEND_HISTOGRAM,
)


class PromClientInterceptor(ClientInterceptor):
    def __init__(
        self,
        enable_client_handling_time_historgram=False,
        enable_client_stream_receive_time_histogram=False,
        enable_client_stream_send_time_histogram=False,
    ):
        self.enable_client_handling_time_historgram = (
            enable_client_handling_time_historgram
        )
        self.enable_client_stream_send_time_histogram = (
            enable_client_stream_send_time_histogram
        )
        self.enable_client_stream_receive_time_histogram = (
            enable_client_stream_receive_time_histogram
        )

    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        client_call_details: ClientCallDetails,
    ):
        _, grpc_service_name, grpc_method_name = parse_method_name(client_call_details)
        handler, _ = get_factory_and_method(method)
        grpc_type = get_method_type(handler)

        if handler.unary_unary:
            GRPC_CLIENT_STARTED_COUNTER.labels(
                grpc_type=grpc_type,
                grpc_service=grpc_service_name,
                grpc_method=grpc_method_name,
            ).inc()

            start = default_timer()
            handler = handler(request_or_iterator, client_call_details)

            if self.enable_client_handling_time_historgram:
                GRPC_CLIENT_HANDLED_HISTORGRAM.labels(
                    grpc_type=grpc_type,
                    grpc_service=grpc_service_name,
                    grpc_method=grpc_method_name,
                ).observe(max(default_timer() - start, 0))

            GRPC_CLIENT_HANDLED_COUNTER.labels(
                grpc_type=grpc_type,
                grpc_service=grpc_service_name,
                grpc_method=grpc_method_name,
                code=handler.code().name,
            ).inc()

        elif handler.unary_stream:
            GRPC_CLIENT_STARTED_COUNTER.labels(
                grpc_type=grpc_type,
                grpc_service=grpc_service_name,
                grpc_method=grpc_method_name,
            ).inc()

            start = default_timer()
            handler = handler(request_or_iterator, client_call_details)

            if self.enable_client_handling_time_historgram:
                GRPC_CLIENT_HANDLED_HISTORGRAM.labels(
                    grpc_type=grpc_type,
                    grpc_service=grpc_service_name,
                    grpc_method=grpc_method_name,
                ).observe(max(default_timer() - start, 0))

            handler = wrap_interator_inc_counter(
                handler,
                GRPC_CLIENT_STREAM_MSG_RECEIVED,
                grpc_type,
                grpc_service_name,
                grpc_method_name,
            )

            if self.enable_client_stream_receive_time_histogram:
                GRPC_CLIENT_STREAM_RECV_HISTOGRAM.labels(
                    grpc_type=grpc_type,
                    grpc_service=grpc_service_name,
                    grpc_method=grpc_method_name,
                ).observe(max(default_timer() - start, 0))

        elif handler.stream_unary:
            iterator_metrics = GRPC_CLIENT_STREAM_MSG_SENT
            request_or_iterator = wrap_interator_inc_counter(
                request_or_iterator,
                iterator_metrics,
                grpc_type,
                grpc_service_name,
                grpc_method_name,
            )

            start = default_timer()
            handler = handler(request_or_iterator, client_call_details)

            GRPC_CLIENT_STARTED_COUNTER.labels(
                grpc_type=grpc_type,
                grpc_service=grpc_service_name,
                grpc_method=grpc_method_name,
            ).inc()
            if self.enable_client_handling_time_historgram:
                GRPC_CLIENT_HANDLED_HISTORGRAM.labels(
                    grpc_type=grpc_type,
                    grpc_service=grpc_service_name,
                    grpc_method=grpc_method_name,
                ).observe(max(default_timer() - start, 0))

            if self.enable_client_stream_send_time_histogram:
                GRPC_CLIENT_STREAM_SEND_HISTOGRAM.labels(
                    grpc_type=grpc_type,
                    grpc_service=grpc_service_name,
                    grpc_method=grpc_method_name,
                ).observe(max(default_timer() - start, 0))

        elif handler.stream_stream:
            start = default_timer()

            iterator_sent_metrc = GRPC_CLIENT_STREAM_MSG_SENT

            response_or_iterator = handler(
                wrap_interator_inc_counter(
                    request_or_iterator,
                    iterator_sent_metrc,
                    grpc_type,
                    grpc_service_name,
                    grpc_method_name,
                ),
                client_call_details,
            )

            if self.enable_client_stream_send_time_histogram:
                GRPC_CLIENT_STREAM_SEND_HISTOGRAM.labels(
                    grpc_type=grpc_type,
                    grpc_service=grpc_service_name,
                    grpc_method=grpc_method_name,
                ).observe(max(default_timer() - start, 0))

            iterator_received_metrics = GRPC_CLIENT_STREAM_MSG_RECEIVED

            response_or_iterator = wrap_interator_inc_counter(
                response_or_iterator,
                iterator_received_metrics,
                grpc_type,
                grpc_service_name,
                grpc_method_name,
            )

            if self.enable_client_stream_receive_time_histogram:
                GRPC_CLIENT_STREAM_RECV_HISTOGRAM.labels(
                    grpc_type=grpc_type,
                    grpc_service=grpc_service_name,
                    grpc_method=grpc_method_name,
                ).observe(max(default_timer() - start, 0))

        else:
            raise RuntimeError("unknown rpc_handler")

        return response_or_iterator if handler.stream_stream else handler
