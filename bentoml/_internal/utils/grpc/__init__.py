from __future__ import annotations

import typing as t
import builtins
import datetime

from google.protobuf.message import Message
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp

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


def datetime_to_timestamp(dt: datetime.datetime) -> Timestamp:
    """
    Convert a datetime object to a Timestamp object.
    """
    timestamp = Timestamp()
    timestamp.FromDatetime(dt)
    return timestamp


def timestamp_to_datetime(timestamp: Timestamp) -> datetime.datetime:
    """
    Convert a Timestamp object to a datetime object.
    """
    return timestamp.ToDatetime()


def timedelta_to_duration(td: datetime.timedelta) -> Duration:
    """
    Convert a timedelta object to a Duration object.
    """
    duration = Duration()
    duration.FromTimedelta(td)
    return duration


def duration_to_timedelta(duration: Duration) -> datetime.timedelta:
    """
    Convert a Duration object to a timedelta object.
    """
    return duration.ToTimedelta()


def repeated(type_callable: builtins.type) -> t.Callable[[list[t.Any]], list[t.Any]]:
    return lambda value: [type_callable(v) for v in value]
