from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from google.protobuf.message import Message

# content-type is always application/grpc
GRPC_CONTENT_TYPE = "application/grpc"


class Codec(ABC):
    _content_subtype: str

    def __new__(cls: type[Self]) -> Self:
        obj = object.__new__(cls)
        if not cls._content_subtype:
            raise TypeError(f"{cls} should have a '_content_subtype' attribute")
        obj.__setattr__("_content_subtype", cls._content_subtype)
        return obj

    @property
    def content_type(self) -> str:
        return self._content_subtype

    @abstractmethod
    def encode(self, message: t.Any, message_type: t.Type[Message]) -> bytes:
        # TODO: We will want to use this to encode headers message.
        pass

    @abstractmethod
    def decode(self, data: bytes, message_type: t.Type[Message]) -> t.Any:
        # TODO: We will want to use this to decode headers message.
        pass


class ProtoCodec(Codec):
    _content_subtype: str = "proto"

    def encode(self, message: t.Any, message_type: t.Type[Message]) -> bytes:
        if not isinstance(message, message_type):
            raise TypeError(f"message should be a {message_type}, got {type(message)}.")
        return message.SerializeToString()

    def decode(self, data: bytes, message_type: t.Type[Message]) -> t.Any:
        return message_type.FromString(data)


def get_grpc_content_type(codec: Codec | None = None) -> str:
    return f"{GRPC_CONTENT_TYPE}" + f"+{codec.content_type}" if codec else ""
