from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.protobuf.message import Message


def proto_to_dict(msg: Message, **kwargs: t.Any) -> dict[str, t.Any]:
    from google.protobuf.json_format import MessageToDict

    if "preserving_proto_field_name" not in kwargs:
        kwargs.setdefault("preserving_proto_field_name", True)

    return MessageToDict(msg, **kwargs)
