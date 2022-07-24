from __future__ import annotations

import enum
from dataclasses import dataclass


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
