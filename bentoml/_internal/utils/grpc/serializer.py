from __future__ import annotations

import builtins

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
