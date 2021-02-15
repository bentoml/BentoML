from typing import Any, Callable, NamedTuple, Tuple

import grpc
from prometheus_client import Counter

UNARY = "UNARY"
SERVER_STREAMING = "SERVER_STREAMING"
CLIENT_STREAMING = "CLIENT_STREAMING"
BIDI_STREAMING = "BIDI_STREAMING"
UNKNOWN = "UNKNOWN"


def wrap_interator_inc_counter(
    iterator: Any,
    counter: Counter,
    grpc_type: str,
    grpc_service_name: str,
    grpc_method_name: str,
):
    for item in iterator:
        counter.labels(
            grpc_type=grpc_type,
            grpc_service=grpc_service_name,
            grpc_method=grpc_method_name,
        ).inc()
        yield item


# get method name for given RPC handler
# UNARY            - both handler.request_streaming and handler.response_streaming=false   - > unary_unary
# SERVER_STREAMING - handler.request_streaming=false and handler.response_streaming=true   - > unary_stream
# CIENT_STREAMING  - handler.request_streaming=true and handler.reqsponse_streaming=false  - > stream_unary
# BIDI_STREAMING   - both handler.request_streaming and handler.response_streaming=true    - > stram_stream
def get_method_type(rpc_handler: grpc.RpcMethodHandler) -> str:
    if rpc_handler.unary_unary:
        return UNARY
    elif rpc_handler.unary_stream:
        return SERVER_STREAMING
    elif rpc_handler.stream_unary:
        return CLIENT_STREAMING
    else:
        return BIDI_STREAMING


def get_factory_and_method(
    rpc_handler: grpc.RpcMethodHandler,
) -> Tuple[Callable, Callable]:
    method_handler, next_handler = None, None
    if rpc_handler.unary_unary:
        method_handler, next_handler = (
            grpc.unary_unary_rpc_method_handler,
            rpc_handler.unary_unary,
        )
    elif rpc_handler.unary_stream:
        method_handler, next_handler = (
            grpc.unary_stream_rpc_method_handler,
            rpc_handler.unary_stream,
        )
    elif rpc_handler.stream_unary:
        method_handler, next_handler = (
            grpc.stream_unary_rpc_method_handler,
            rpc_handler.stream_unary,
        )
    elif rpc_handler.stream_stream:
        method_handler, next_handler = (
            grpc.stream_stream_rpc_method_handler,
            rpc_handler.stream_stream,
        )
    return method_handler, next_handler


class MethodName(NamedTuple):
    """represents a gRPC handler call details
    attr: /packages.services/method"""

    package_name: str
    service_name: str
    method_name: str

    @property
    def get_services(self):
        return (
            f"{self.package_name}.{self.service_name}"
            if self.package_name
            else self.service_name
        )


def parse_method_name(method_name: str) -> MethodName:
    """
    Parse method name into packages - services - method

    example: /bentoml.Yatai/HealthCheck
    -> returns methods: bentoml
               services: Yatai
               endpoint: HealthCheck
    """
    _, package_service, method = method_name.split("/")
    # TODO: can packages name be bentoml.pkg1.pkg2?
    *tmp_packages, service = package_service.rsplit(".", maxsplit=1)
    package = tmp_packages[0] if tmp_packages else ""
    return MethodName(package, service, method)
