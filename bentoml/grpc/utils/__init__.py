from __future__ import annotations

import enum
import typing as t
import logging
from http import HTTPStatus
from typing import overload
from typing import TYPE_CHECKING
from dataclasses import dataclass

from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import UnprocessableEntity
from bentoml.grpc.utils.mapping import grpc_status_to_http_status_map
from bentoml.grpc.utils.mapping import http_status_to_grpc_status_map
from bentoml._internal.utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import grpc
    from grpc import aio
    from google.protobuf.struct_pb2 import Value

    from bentoml.io import File
    from bentoml.io import JSON
    from bentoml.io import Text
    from bentoml.io import Image
    from bentoml.io import Multipart
    from bentoml.io import IODescriptor
    from bentoml.io import NumpyNdarray
    from bentoml.io import PandasSeries
    from bentoml.io import PandasDataFrame
    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.types import MessageType
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import BentoServicerContext
else:
    exc_msg = "'grpc' is required. Install with 'pip install grpcio'."
    grpc = LazyLoader("grpc", globals(), "grpc", exc_msg=exc_msg)
    aio = LazyLoader("aio", globals(), "grpc.aio", exc_msg=exc_msg)
    pb = LazyLoader("pb", globals(), "bentoml.grpc.v1.service_pb2")

__all__ = [
    "grpc_status_code",
    "parse_method_name",
    "to_http_status",
    "get_field",
    "serialize_proto",
    "raise_grpc_exception",
    "GRPC_CONTENT_TYPE",
    "validate_content_type",
]

logger = logging.getLogger(__name__)


# content-type is always application/grpc
GRPC_CONTENT_TYPE = "application/grpc"


def validate_content_type(
    context: BentoServicerContext, descriptor: IODescriptor[t.Any]
) -> None:
    """
    Validate 'content-type' from invocation metadata.
    """
    metadata = context.invocation_metadata()
    if metadata:
        if TYPE_CHECKING:
            from grpc.aio._typing import MetadatumType

            metadata = t.cast(tuple[MetadatumType], metadata)

        metas = aio.Metadata.from_tuple(metadata)
        maybe_content_type = metas.get_all("content-type")
        if maybe_content_type:
            if len(maybe_content_type) > 1:
                raise_grpc_exception(
                    f"{maybe_content_type} should only contain one 'Content-Type' headers.",
                    context=context,
                    exception_cls=InvalidArgument,
                )

            content_type = str(maybe_content_type[0])

            if not content_type.startswith(GRPC_CONTENT_TYPE):
                raise_grpc_exception(
                    f"{content_type} should startwith {GRPC_CONTENT_TYPE}.",
                    context=context,
                    exception_cls=InvalidArgument,
                )
            if content_type != descriptor.grpc_content_type:
                raise_grpc_exception(
                    f"'{content_type}' is found while '{repr(descriptor)}' requires '{descriptor.grpc_content_type}'.",
                    context=context,
                    exception_cls=InvalidArgument,
                )


@overload
def get_field(req: pb.Request, descriptor: File) -> MessageType[pb.File]:
    ...


@overload
def get_field(req: pb.Request, descriptor: Image) -> MessageType[pb.File]:
    ...


@overload
def get_field(req: pb.Request, descriptor: JSON) -> MessageType[Value]:
    ...


@overload
def get_field(
    req: pb.Request, descriptor: Multipart
) -> MessageType[dict[str, pb.Part]]:
    ...


@overload
def get_field(req: pb.Request, descriptor: NumpyNdarray) -> MessageType[pb.NDArray]:
    ...


@overload
def get_field(
    req: pb.Request, descriptor: PandasDataFrame
) -> MessageType[pb.DataFrame]:
    ...


@overload
def get_field(req: pb.Request, descriptor: PandasSeries) -> MessageType[pb.Series]:
    ...


@overload
def get_field(req: pb.Request, descriptor: Text) -> MessageType[str]:
    ...


def get_field(req: pb.Request, descriptor: IODescriptor[t.Any]) -> MessageType[t.Any]:
    try:
        _ = req.HasField(descriptor.proto_field)
    except KeyError as e:
        raise UnprocessableEntity(
            f"Missing required '{descriptor.proto_field}' for {descriptor.__class__.__name__}.: {str(e)}"
        ) from e
    return getattr(req, descriptor.proto_field)


def serialize_proto(output: dict[str, t.Any], **kwargs: t.Any) -> pb.Response:
    from google.protobuf.json_format import ParseDict

    return ParseDict(output, pb.Response(), **kwargs)


def raise_grpc_exception(
    msg: str,
    context: BentoServicerContext,
    exception_cls: t.Type[BentoMLException] = BentoMLException,
):
    code = http_status_to_grpc_status_map().get(
        exception_cls.error_code, grpc.StatusCode.UNKNOWN
    )
    context.set_code(code)
    context.set_details(msg)
    raise exception_cls(msg)


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
    # are set, then it is guaranteed that one of the RpcMethodHandler are not None.
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
