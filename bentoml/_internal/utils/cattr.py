from __future__ import annotations

import typing as t
from datetime import datetime

from .pkg import pkg_version_info

cattr_major_minor = pkg_version_info("cattrs")
if cattr_major_minor[:2] <= (22, 2):
    from cattr import GenConverter as Converter
else:
    from cattr import Converter

from attr import fields
from cattr import override  # type: ignore
from cattr.gen import AttributeOverride

bentoml_cattr = Converter(forbid_extra_keys=True)


def omit_if_init_false(cls: t.Any) -> dict[str, AttributeOverride]:
    return {f.name: override(omit=True) for f in fields(cls) if not f.init}


def omit_if_default(cls: t.Any) -> dict[str, AttributeOverride]:
    return {f.name: override(omit_if_default=True) for f in fields(cls)}


def datetime_structure_hook(dt_like: str | datetime, _: t.Any) -> datetime:
    if isinstance(dt_like, str):
        return datetime.fromisoformat(dt_like)
    elif isinstance(dt_like, datetime):
        return dt_like
    else:
        raise Exception(f"Unable to parse datetime from '{dt_like}'")


bentoml_cattr.register_structure_hook(datetime, datetime_structure_hook)
bentoml_cattr.register_unstructure_hook(datetime, lambda dt: dt.isoformat())
