from __future__ import annotations

import enum
import typing as t
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING
from dataclasses import dataclass

import grpc

from bentoml.exceptions import BentoMLException

from .serializer import proto_to_dict

if TYPE_CHECKING:
    from ...server.grpc.types import RequestType
    from ...server.grpc.types import ResponseType
    from ...server.grpc.types import RpcMethodHandler
    from ...server.grpc.types import BentoServicerContext

__all__ = [
    "grpc_status_code",
    "parse_method_name",
    "get_method_type",
    "get_factory_and_method",
    "proto_to_dict",
]

logger = logging.getLogger(__name__)

_STATUS_CODE_MAPPING = {
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
    HTTPStatus.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    HTTPStatus.UNPROCESSABLE_ENTITY: grpc.StatusCode.FAILED_PRECONDITION,
}


def grpc_status_code(err: BentoMLException) -> grpc.StatusCode:
    """
    Convert BentoMLException.error_code to grpc.StatusCode.
    """
    return _STATUS_CODE_MAPPING.get(err.error_code, grpc.StatusCode.UNKNOWN)


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
        package: This is defined by `package foo.bar`,
        designation in the protocol buffer definition
        service: service name in protocol buffer
        definition (eg: service SearchService { ... })
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


def get_method_type(request_streaming: bool, response_streaming: bool) -> str:
    if not request_streaming and not response_streaming:
        return RpcMethodType.UNARY
    elif not request_streaming and response_streaming:
        return RpcMethodType.SERVER_STREAMING
    elif request_streaming and not response_streaming:
        return RpcMethodType.CLIENT_STREAMING
    elif request_streaming and response_streaming:
        return RpcMethodType.BIDI_STREAMING
    else:
        return RpcMethodType.UNKNOWN


def get_factory_and_method(
    rpc_handler: RpcMethodHandler,
) -> tuple[
    t.Callable[..., t.Any],
    t.Callable[[RequestType, BentoServicerContext], t.Awaitable[ResponseType]],
]:
    if rpc_handler.unary_unary:
        return grpc.unary_unary_rpc_method_handler, rpc_handler.unary_unary
    elif rpc_handler.unary_stream:
        return grpc.unary_stream_rpc_method_handler, rpc_handler.unary_stream
    elif rpc_handler.stream_unary:
        return grpc.stream_unary_rpc_method_handler, rpc_handler.stream_unary
    elif rpc_handler.stream_stream:
        return grpc.stream_stream_rpc_method_handler, rpc_handler.stream_stream
    else:
        raise BentoMLException(f"RPC method handler {rpc_handler} does not exist.")
