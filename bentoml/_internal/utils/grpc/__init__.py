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

from ..lazy_loader import LazyLoader

if TYPE_CHECKING:

    from bentoml.io import IODescriptor
    from bentoml.grpc.v1 import service_pb2

    from ...server.grpc.types import HandlerMethod
    from ...server.grpc.types import HandlerFactoryFn
    from ...server.grpc.types import RpcMethodHandler

    # keep sync with bentoml.grpc.v1.service.Response
    ContentsDict = dict[str, dict[str, t.Any]]
else:
    service_pb2 = LazyLoader("service_pb2", globals(), "bentoml.grpc.v1.service_pb2")

__all__ = [
    "grpc_status_code",
    "parse_method_name",
    "get_method_type",
    "get_rpc_handler",
    "deserialize_proto",
    "serialize_proto",
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

    kind = req.contents.WhichOneof("kind")
    if kind not in io_descriptor.accepted_proto_kind:
        raise UnprocessableEntity(
            f"{kind} is not supported for {io_descriptor.__class__.__name__}. Supported message fields are: {io_descriptor.accepted_proto_kind}"
        )

    return kind, MessageToDict(getattr(req.contents, kind), **kwargs)


def serialize_proto(fields: str, contents_dict: ContentsDict) -> service_pb2.Response:
    from google.protobuf.json_format import ParseDict

    return ParseDict({"contents": {fields: contents_dict}}, service_pb2.Response())


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


def get_rpc_handler(
    handler: RpcMethodHandler,
) -> tuple[HandlerFactoryFn, HandlerMethod[t.Any]]:
    if handler.unary_unary:
        return grpc.unary_unary_rpc_method_handler, handler.unary_unary
    elif handler.unary_stream:
        return grpc.unary_stream_rpc_method_handler, handler.unary_stream
    elif handler.stream_unary:
        return grpc.stream_unary_rpc_method_handler, handler.stream_unary
    elif handler.stream_stream:
        return grpc.stream_stream_rpc_method_handler, handler.stream_stream
    else:
        raise BentoMLException(f"RPC method handler {handler} does not exist.")


def invoke_handler_factory(
    fn: HandlerMethod[t.Any], factory: HandlerFactoryFn, handler: RpcMethodHandler
) -> t.Any:
    return factory(
        fn,
        request_deserializer=handler.request_deserializer,
        response_serializer=handler.response_serializer,
    )
