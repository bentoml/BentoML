from __future__ import annotations

import io
import types
import typing as t

from typing_extensions import Literal

LITERAL_TYPES: set[type] = {Literal}
if hasattr(t, "Literal"):
    LITERAL_TYPES.add(t.Literal)
BINARY_FILE_TYPES: set[type] = {t.BinaryIO, t.IO[bytes], io.BytesIO}
LIST_TYPES: set[type] = {list, t.List, t.Sequence, t.MutableSequence}
TUPLE_TYPES: set[type] = {tuple, t.Tuple}


def get_origin(type_: t.Any) -> type:
    return t.get_origin(type_) or type_


def is_literal_type(typ_: t.Any) -> bool:
    return get_origin(typ_) in LITERAL_TYPES


def is_binary_file_type(typ_: t.Any) -> bool:
    return typ_ in BINARY_FILE_TYPES


def is_list_type(typ_: t.Any) -> bool:
    origin = get_origin(typ_)
    return issubclass(origin, list) or origin in LIST_TYPES


def is_tuple_type(typ_: t.Any) -> bool:
    return get_origin(typ_) in TUPLE_TYPES


def is_union_type(typ_: t.Any) -> bool:
    return get_origin(typ_) in (t.Union, types.UnionType)
