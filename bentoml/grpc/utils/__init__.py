from __future__ import annotations

import typing as t
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING
from functools import lru_cache
from dataclasses import dataclass

from bentoml.exceptions import InvalidArgument

if TYPE_CHECKING:
    import types
    from enum import Enum

    import grpc

    from bentoml.exceptions import BentoMLException
    from bentoml.grpc.types import ProtoField
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import BentoServicerContext
    from bentoml.grpc.v1alpha1 import service_pb2 as pb
    from bentoml._internal.io_descriptors import IODescriptor

    # We need this here so that __all__ is detected due to lazy import
    def import_generated_stubs(
        version: str = "v1alpha1",
    ) -> tuple[types.ModuleType, types.ModuleType]:
        ...

    def import_grpc() -> tuple[types.ModuleType, types.ModuleType]:
        ...

else:
    from bentoml.grpc.utils._import_hook import import_grpc
    from bentoml.grpc.utils._import_hook import import_generated_stubs

    pb, _ = import_generated_stubs()
    grpc, _ = import_grpc()

__all__ = [
    "grpc_status_code",
    "parse_method_name",
    "to_http_status",
    "GRPC_CONTENT_TYPE",
    "import_generated_stubs",
    "import_grpc",
    "validate_proto_fields",
]

logger = logging.getLogger(__name__)

# content-type is always application/grpc
GRPC_CONTENT_TYPE = "application/grpc"


def validate_proto_fields(
    field: str | None, io_: IODescriptor[t.Any]
) -> str | ProtoField:
    if field is None:
        raise InvalidArgument('"field" cannot be empty.')
    accepted_fields = io_._proto_fields + ("serialized_bytes",)
    if field not in accepted_fields:
        raise InvalidArgument(
            f"'{io_.__class__.__name__}' accepts one of the following fields: '{','.join(accepted_fields)}' got '{field}' instead.",
        ) from None
    return field


@lru_cache(maxsize=1)
def http_status_to_grpc_status_map() -> dict[Enum, grpc.StatusCode]:
    # Maps HTTP status code to grpc.StatusCode
    from http import HTTPStatus

    return {
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


@lru_cache(maxsize=1)
def grpc_status_to_http_status_map() -> dict[grpc.StatusCode, Enum]:
    return {v: k for k, v in http_status_to_grpc_status_map().items()}


@lru_cache(maxsize=1)
def filetype_pb_to_mimetype_map() -> dict[pb.File.FileType.ValueType, str]:
    return {
        pb.File.FILE_TYPE_CSV: "text/csv",
        pb.File.FILE_TYPE_PLAINTEXT: "text/plain",
        pb.File.FILE_TYPE_JSON: "application/json",
        pb.File.FILE_TYPE_BYTES: "application/octet-stream",
        pb.File.FILE_TYPE_PDF: "application/pdf",
        pb.File.FILE_TYPE_PNG: "image/png",
        pb.File.FILE_TYPE_JPEG: "image/jpeg",
        pb.File.FILE_TYPE_GIF: "image/gif",
        pb.File.FILE_TYPE_TIFF: "image/tiff",
        pb.File.FILE_TYPE_BMP: "image/bmp",
        pb.File.FILE_TYPE_WEBP: "image/webp",
        pb.File.FILE_TYPE_SVG: "image/svg+xml",
    }


@lru_cache(maxsize=1)
def mimetype_to_filetype_pb_map() -> dict[str, pb.File.FileType.ValueType]:
    return {v: k for k, v in filetype_pb_to_mimetype_map().items()}


def grpc_status_code(err: BentoMLException) -> grpc.StatusCode:
    """
    Convert BentoMLException.error_code to grpc.StatusCode.
    """
    return http_status_to_grpc_status_map().get(err.error_code, grpc.StatusCode.UNKNOWN)


def to_http_status(status_code: grpc.StatusCode) -> int:
    """
    Convert grpc.StatusCode to HTTPStatus.
    """
    status = grpc_status_to_http_status_map().get(
        status_code, HTTPStatus.INTERNAL_SERVER_ERROR
    )

    return status.value


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
    method = method_name.split("/", maxsplit=2)
    # sanity check for method.
    if len(method) != 3:
        return MethodName(), False
    _, package_service, method = method
    *packages, service = package_service.rsplit(".", maxsplit=1)
    package = packages[0] if packages else ""
    return MethodName(package, service, method), True


def wrap_rpc_handler(
    wrapper: t.Callable[
        ...,
        t.Callable[
            [pb.Request, BentoServicerContext],
            t.Coroutine[t.Any, t.Any, pb.Response | t.Awaitable[pb.Response]],
        ],
    ],
    handler: RpcMethodHandler | None,
) -> RpcMethodHandler | None:
    if not handler:
        return None
    if not handler.request_streaming and not handler.response_streaming:
        assert handler.unary_unary
        return handler._replace(unary_unary=wrapper(handler.unary_unary))
    elif not handler.request_streaming and handler.response_streaming:
        assert handler.unary_stream
        return handler._replace(unary_stream=wrapper(handler.unary_stream))
    elif handler.request_streaming and not handler.response_streaming:
        assert handler.stream_unary
        return handler._replace(stream_unary=wrapper(handler.stream_unary))
    elif handler.request_streaming and handler.response_streaming:
        assert handler.stream_stream
        return handler._replace(stream_stream=wrapper(handler.stream_stream))
    else:
        raise RuntimeError(f"RPC method handler {handler} does not exist.") from None
