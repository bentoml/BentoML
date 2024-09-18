from __future__ import annotations

import typing as t
from datetime import datetime

import cattr
from dateutil.parser import parse

from bentoml._internal.tag import Tag

time_format = "%Y-%m-%d %H:%M:%S.%f"
T = t.TypeVar("T")


def datetime_encoder(time_obj: t.Optional[datetime]) -> t.Optional[str]:
    if not time_obj:
        return None
    return time_obj.strftime(time_format)


def datetime_decoder(datetime_str: t.Optional[str], _: t.Any) -> t.Optional[datetime]:
    if not datetime_str:
        return None
    return parse(datetime_str)


def tag_encoder(tag_obj: t.Optional[Tag]) -> t.Optional[str]:
    if not tag_obj:
        return None
    return str(tag_obj)


def tag_decoder(tag_str: t.Optional[str], _: t.Any) -> t.Optional[Tag]:
    if not tag_str:
        return None
    return Tag.from_str(tag_str)


def dict_options_converter(
    options_type: type[T],
) -> t.Callable[[T | dict[str, T]], T]:
    def _converter(value: T | dict[str, T] | None) -> T:
        if value is None:
            return options_type()
        if isinstance(value, dict):
            return options_type(**value)
        return value

    return _converter


cloud_converter = cattr.Converter()

cloud_converter.register_unstructure_hook(datetime, datetime_encoder)
cloud_converter.register_structure_hook(datetime, datetime_decoder)
cloud_converter.register_unstructure_hook(Tag, tag_encoder)
cloud_converter.register_structure_hook(Tag, tag_decoder)


def schema_to_object(obj: t.Any) -> t.Any:
    return cloud_converter.unstructure(obj, obj.__class__)


def schema_from_object(obj: t.Any, cls: t.Type[T]) -> T:
    return cloud_converter.structure(obj, cls)
