from __future__ import annotations

import builtins
from dataclasses import dataclass

from google.protobuf.message import Message
from google.protobuf.descriptor import FieldDescriptor

# defines a descriptor type to python native type.
TYPE_CALLABLE_MAP: dict[int, type] = {
    FieldDescriptor.TYPE_DOUBLE: builtins.float,
    FieldDescriptor.TYPE_FLOAT: builtins.float,
    FieldDescriptor.TYPE_INT32: builtins.int,
    FieldDescriptor.TYPE_INT64: builtins.int,
    FieldDescriptor.TYPE_UINT32: builtins.int,
    FieldDescriptor.TYPE_UINT64: builtins.int,
    FieldDescriptor.TYPE_SINT32: builtins.int,
    FieldDescriptor.TYPE_SINT64: builtins.int,
    FieldDescriptor.TYPE_FIXED32: builtins.int,
    FieldDescriptor.TYPE_FIXED64: builtins.int,
    FieldDescriptor.TYPE_SFIXED32: builtins.int,
    FieldDescriptor.TYPE_SFIXED64: builtins.int,
    FieldDescriptor.TYPE_BOOL: builtins.bool,
    FieldDescriptor.TYPE_STRING: builtins.str,
    FieldDescriptor.TYPE_BYTES: builtins.bytes,
    FieldDescriptor.TYPE_ENUM: builtins.int,
}


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
