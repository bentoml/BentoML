from __future__ import annotations

import typing as t
from datetime import datetime

import cattr
from attr import fields
from cattr import override  # type: ignore
from cattr.gen import AttributeOverride

# from collections import Set


# TODO: migrate to cattr.GenConverter for better performance, e.g.:
# bentoml_cattr = cattr.GenConverter(
#     unstruct_collection_overrides={tuple: list, Set: list},
#     omit_if_default=True,
# )
bentoml_cattr = cattr.Converter()

# Resolve any forward references during (de)serialization
# bentoml_cattr.register_unstructure_hook_func(
#     lambda cls: cls.__class__ is typing.ForwardRef,
#     lambda obj, cls=None: bentoml_cattr.unstructure(
#         obj, cls.__forward_value__ if cls else None
#     ),
# )
# bentoml_cattr.register_structure_hook_func(
#     lambda cls: cls.__class__ is typing.ForwardRef,
#     lambda obj, cls: bentoml_cattr.structure(obj, cls.__forward_value__),
# )


def omit_if_init_false(cls: t.Any) -> dict[str, AttributeOverride]:
    return {f.name: override(omit=True) for f in fields(cls) if not f.init}


def omit_if_default(cls: t.Any) -> dict[str, AttributeOverride]:
    return {f.name: override(omit_if_default=True) for f in fields(cls)}


def datetime_decoder(dt_like: str | datetime, _: t.Any) -> datetime:
    if isinstance(dt_like, str):
        return datetime.fromisoformat(dt_like)
    elif isinstance(dt_like, datetime):
        return dt_like
    else:
        raise Exception(f"Unable to parse datetime from '{dt_like}'")


bentoml_cattr.register_unstructure_hook(datetime, lambda dt: dt.isoformat())
bentoml_cattr.register_structure_hook(datetime, datetime_decoder)
