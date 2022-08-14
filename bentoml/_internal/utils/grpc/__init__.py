from __future__ import annotations

import enum
import typing as t
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING
from dataclasses import dataclass

import grpc

from bentoml.exceptions import BentoMLException
from bentoml.exceptions import UnprocessableEntity

from .codec import ProtoCodec
from .codec import get_grpc_content_type
from ..lazy_loader import LazyLoader

if TYPE_CHECKING:

    from bentoml.io import IODescriptor
    from bentoml.grpc.v1 import service_pb2

    from ...server.grpc.types import RpcMethodHandler
else:
    service_pb2 = LazyLoader("service_pb2", globals(), "bentoml.grpc.v1.service_pb2")

__all__ = [
    "grpc_status_code",
    "parse_method_name",
    "deserialize_proto",
    "to_http_status",
    "get_grpc_content_type",
    "ProtoCodec",
]

logger = logging.getLogger(__name__)


def deserialize_proto(
    io_descriptor: IODescriptor[t.Any],
    req: service_pb2.Request,
    **kwargs: t.Any,
) -> tuple[str, dict[str, t.Any]]:
    # Deserialize a service_pb2.Request to dict.
    from google.protobuf.json_format import MessageToDict

    if "preserving_proto_field_name" not in kwargs:
        kwargs.setdefault("preserving_proto_field_name", True)

    kind = req.input.WhichOneof("kind")
    if kind not in io_descriptor.accepted_proto_fields:
        raise UnprocessableEntity(
            f"{kind} is not supported for {io_descriptor.__class__.__name__}. Supported protobuf message fields are: {io_descriptor.accepted_proto_fields}"
        )

    return kind, MessageToDict(getattr(req.input, kind), **kwargs)


# Maps HTTP status code to grpc.StatusCode
_STATUS_CODE_MAPPING = {
    HTTPStatus.OK: grpc.StatusCode.OK,
    HTTPStatus.UNAUTHORIZED: grpc.StatusCode.UNAUTHENTICATED,
    HTTPStatus.FORBIDDEN: grpc.StatusCode.PERMISSION_DENIED,
    HTTPStatus.NOT_FOUND: grpc.StatusCode.UNIMPLEMENTED,
    HTTPStatus.TOO_MANY_REQUESTS: grpc.StatusCode.UNAVAILABLE,
    HTTPStatus.BAD_GATEWAY: grpc.StatusCode.UNAVAILABLE,
    HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT: grpc.StatusCode.DEADLINE_EXCEEDED,
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
    HTTPStatus.UNPROCESSABLE_ENTITY: grpc.StatusCode.FAILED_PRECONDITION,
}


def grpc_status_code(err: BentoMLException) -> grpc.StatusCode:
    """
    Convert BentoMLException.error_code to grpc.StatusCode.
    """
    return _STATUS_CODE_MAPPING.get(err.error_code, grpc.StatusCode.UNKNOWN)


def to_http_status(status_code: grpc.StatusCode) -> int:
    """
    Convert grpc.StatusCode to HTTPStatus.
    """
    try:
        status = {v: k for k, v in _STATUS_CODE_MAPPING.items()}[status_code]
    except KeyError:
        status = HTTPStatus.INTERNAL_SERVER_ERROR

    return status.value


class RpcMethodType(str, enum.Enum):
    UNARY = "UNARY"
    CLIENT_STREAMING = "CLIENT_STREAMING"
    SERVER_STREAMING = "SERVER_STREAMING"
    BIDI_STREAMING = "BIDI_STREAMING"
    UNKNOWN = "UNKNOWN"


@dataclass
class MethodName:
    """
    Represents a gRPC method name.

    Attributes:
        package: This is defined by `package foo.bar`, designation in the protocol buffer definition
        service: service name in protocol buffer definition (eg: service SearchService { ... })
        method: method name
    """

    package: str = ""
    service: str = ""
    method: str = ""

    @property
    def fully_qualified_service(self):
        """return the service name prefixed with package"""
        return f"{self.package}.{self.service}" if self.package else self.service


def parse_method_name(method_name: str) -> tuple[MethodName, bool]:
    """
    Infers the grpc service and method name from the handler_call_details.
    e.g. /package.ServiceName/MethodName
    """
    if len(method_name.split("/")) < 3:
        return MethodName(), False
    _, package_service, method = method_name.split("/")
    *packages, service = package_service.rsplit(".", maxsplit=1)
    package = packages[0] if packages else ""
    return MethodName(package, service, method), True


def wrap_rpc_handler(
    wrapper: t.Callable[..., t.Any],
    handler: RpcMethodHandler | None,
) -> RpcMethodHandler | None:
    if not handler:
        return None

    # The reason we are using TYPE_CHECKING for assert here
    # is that if the following bool request_streaming and response_streaming
    # are set, then it is guaranteed that RpcMethodHandler are not None.
    if not handler.request_streaming and not handler.response_streaming:
        if TYPE_CHECKING:
            assert handler.unary_unary
        return handler._replace(unary_unary=wrapper(handler.unary_unary))
    elif not handler.request_streaming and handler.response_streaming:
        if TYPE_CHECKING:
            assert handler.unary_stream
        return handler._replace(unary_stream=wrapper(handler.unary_stream))
    elif handler.request_streaming and not handler.response_streaming:
        if TYPE_CHECKING:
            assert handler.stream_unary
        return handler._replace(stream_unary=wrapper(handler.stream_unary))
    elif handler.request_streaming and handler.response_streaming:
        if TYPE_CHECKING:
            assert handler.stream_stream
        return handler._replace(stream_stream=wrapper(handler.stream_stream))
    else:
        raise BentoMLException(f"RPC method handler {handler} does not exist.")
