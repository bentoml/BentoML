from __future__ import annotations

import typing as t
from datetime import datetime

import cattr
from attr import fields
from cattr import override  # type: ignore
from cattr.gen import AttributeOverride

bentoml_cattr = cattr.Converter()


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


bentoml_cattr.register_unstructure_hook(datetime, lambda dt: dt.isoformat())  # type: ignore
bentoml_cattr.register_structure_hook(datetime, datetime_decoder)
