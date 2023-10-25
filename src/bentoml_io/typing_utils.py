from __future__ import annotations

import collections.abc
import types
import typing as t

from typing_extensions import Literal

if t.TYPE_CHECKING:
    from PIL import Image

LITERAL_TYPES: set[type] = {Literal}
if hasattr(t, "Literal"):
    LITERAL_TYPES.add(t.Literal)
LIST_TYPES: set[type] = {list, t.List, t.Sequence, t.MutableSequence}
TUPLE_TYPES: set[type] = {tuple, t.Tuple}
SYNC_ITERATOR_TYPES: set[type] = {
    t.Iterator,
    t.Generator,
    collections.abc.Iterator,
    collections.abc.Generator,
}

ASYNC_ITERATOR_TYPES: set[type] = {
    t.AsyncIterator,
    t.AsyncGenerator,
    collections.abc.AsyncIterator,
    collections.abc.AsyncGenerator,
}


def get_origin(type_: t.Any) -> type:
    return t.get_origin(type_) or type_


def is_literal_type(typ_: t.Any) -> bool:
    return get_origin(typ_) in LITERAL_TYPES


def is_list_type(typ_: t.Any) -> bool:
    origin = get_origin(typ_)
    return issubclass(origin, list) or origin in LIST_TYPES


def is_tuple_type(typ_: t.Any) -> bool:
    return get_origin(typ_) in TUPLE_TYPES


def is_union_type(typ_: t.Any) -> bool:
    return get_origin(typ_) in (t.Union, types.UnionType)


def is_iterator_type(typ_: t.Any) -> bool:
    return get_origin(typ_) in (SYNC_ITERATOR_TYPES | ASYNC_ITERATOR_TYPES)


def is_file_like(obj: t.Any) -> t.TypeGuard[t.BinaryIO]:
    return hasattr(obj, "read") and hasattr(obj, "seek")


def is_image_type(type_: type) -> t.TypeGuard[type[Image.Image]]:
    from bentoml._internal.io_descriptors.image import PIL

    return type_.__module__.startswith("PIL.") and issubclass(type_, PIL.Image.Image)
