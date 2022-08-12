from __future__ import annotations

import enum
import typing as t
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING
from dataclasses import dataclass

import grpc

from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import UnprocessableEntity

from ..lazy_loader import LazyLoader

if TYPE_CHECKING:

    from bentoml.io import IODescriptor
    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.types import MessageType
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import BentoServicerContext
else:
    pb = LazyLoader("pb", globals(), "bentoml.grpc.v1.service_pb2")

__all__ = [
    "grpc_status_code",
    "parse_method_name",
    "deserialize_proto",
    "to_http_status",
    "get_field",
    "serialize_proto",
    "raise_grpc_exception",
    "get_grpc_content_type",
    "GRPC_CONTENT_TYPE",
    "validate_content_type",
    "VALUES_TO_NP_DTYPE_MAP",
]

logger = logging.getLogger(__name__)


# content-type is always application/grpc
GRPC_CONTENT_TYPE = "application/grpc"


# TODO: support the following types for for protobuf message:
# - support complex64, complex128, object and struct types
# - BFLOAT16, QINT32, QINT16, QUINT16, QINT8, QUINT8
#
# For int16, uint16, int8, uint8 -> specify types in NumpyNdarray + using int_values.
#
# For bfloat16, half (float16) -> specify types in NumpyNdarray + using float_values.
#
# for string_values, use <U for np.dtype instead of S (zero-terminated bytes).
VALUES_TO_NP_DTYPE_MAP = {
    "bool_values": "bool",
    "float_values": "float32",
    "string_values": "<U",
    "double_values": "float64",
    "int_values": "int32",
    "long_values": "int64",
    "uint32_values": "uint32",
    "uint64_values": "uint64",
}


def validate_content_type(
    context: BentoServicerContext, descriptor: IODescriptor[t.Any]
) -> None:
    metadata = context.invocation_metadata()
    if metadata:
        maybe_content_type = metadata.get_all("content-type")
        if maybe_content_type:
            maybe_content_type = list(map(str, maybe_content_type))
            if len(maybe_content_type) > 1:
                raise_grpc_exception(
                    f"{maybe_content_type} should only contain one 'Content-Type' headers.",
                    context=context,
                    exc_cls=InvalidArgument,
                )

            content_type = maybe_content_type[0]
            rpc_content_type = (
                f"{GRPC_CONTENT_TYPE}+{descriptor.mimetype.split('/')[-1]}"
            )

            if not content_type.startswith(GRPC_CONTENT_TYPE):
                raise_grpc_exception(
                    f"{content_type} should startwith {GRPC_CONTENT_TYPE}.",
                    context=context,
                    exc_cls=InvalidArgument,
                )
            if content_type != rpc_content_type:
                raise_grpc_exception(
                    f"{descriptor.__class__.__name__} sets Content-Type '{rpc_content_type}', got {content_type} instead",
                    context=context,
                    exc_cls=BentoMLException,
                )


def get_grpc_content_type(message_format: str | None = None) -> str:
    return f"{GRPC_CONTENT_TYPE}" + f"+{message_format}" if message_format else ""


def get_field(req: pb.Request, descriptor: IODescriptor[t.Any]) -> MessageType[t.Any]:
    if not req.HasField(descriptor.proto_field):
        raise UnprocessableEntity(
            f"Missing required '{descriptor.proto_field}' for {descriptor.__class__.__name__}."
        )
    return getattr(req, descriptor.proto_field)


def deserialize_proto(req: pb.Request, **kwargs: t.Any) -> DeserializeDict:
    # Deserialize a pb.Request to dict.
    from google.protobuf.json_format import MessageToDict

    if "preserving_proto_field_name" not in kwargs:
        kwargs.setdefault("preserving_proto_field_name", True)

    return t.cast("DeserializeDict", MessageToDict(req.input, **kwargs))


def serialize_proto(output: SerializeDict, **kwargs: t.Any) -> pb.Response:
    from google.protobuf.json_format import ParseDict

    return ParseDict({"output": output}, pb.Response(), **kwargs)


def raise_grpc_exception(
    msg: str, context: BentoServicerContext, exc_cls: t.Type[BentoMLException]
):
    context.set_code(
        _STATUS_CODE_MAPPING.get(exc_cls.error_code, grpc.StatusCode.UNKNOWN)
    )
    context.set_details(msg)
    raise exc_cls(msg)


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
